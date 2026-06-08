//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// macro para comprobar errores de CUDA facilmente
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

// tamaño del tile para los kernels (32x32 = 1024 threads por bloque)
const int TILE_WIDTH=32;
const int TILE_HEIGHT=32;

// tamaño del filtro, se cambia en create_filter segun el filtro elegido
int FILTERSIZE=5;

// maximo radio del filtro (para filtro 9x9, halfWidth=4)
#define MAX_HALF_WIDTH 4
// maximo numero de elementos en el filtro (9x9=81)
#define MAX_FILTER_SIZE 81
// filtro en memoria de constantes (cache rapida, solo lectura)
__constant__ float d_filter_const[MAX_FILTER_SIZE];

// comentar/descomentar para activar cada funcionalidad
#define _CONSTANT_MEMORY
#define _SHARED_MEMORY
#define _CANNY_EDGE

// convierte imagen RGBA a escala de grises
__global__
void grayscale(const uchar4* const inputRGBA, float* outputGray,
    int numRows, int numCols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= numCols || y >= numRows) return;

    int idx = y * numCols + x;
    uchar4 pixel = inputRGBA[idx];
    // formula estandar de luminancia
    outputGray[idx] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
}

// calcula magnitud y direccion del gradiente con Sobel
__global__
void gradient_magnitude_direction(const float* input, float* magnitude, float* direction,
    int numRows, int numCols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= numCols || y >= numRows) return;

    // kernels de Sobel horizontal y vertical
    float sobelH[9] = { -1,0,1, -2,0,2, -1,0,1 };
    float sobelV[9] = { -1,-2,-1, 0,0,0, 1,2,1 };

    float Gx = 0, Gy = 0;
    for (int fy = -1; fy <= 1; fy++) {
        for (int fx = -1; fx <= 1; fx++) {
            int nx = x + fx, ny = y + fy;
            float val = 0.0f;
            // si el vecino esta dentro de la imagen, cogemos su valor
            if (nx >= 0 && nx < numCols && ny >= 0 && ny < numRows)
                val = input[ny * numCols + nx];
            int fi = (fy + 1) * 3 + (fx + 1);
            Gx += sobelH[fi] * val;
            Gy += sobelV[fi] * val;
        }
    }

    int idx = y * numCols + x;
    magnitude[idx] = sqrtf(Gx * Gx + Gy * Gy);
    direction[idx] = atan2f(Gy, Gx);
}

// suprime pixels que no son maximo local en la direccion del gradiente
__global__
void non_max_suppression(const float* magnitude, const float* direction,
    float* output, int numRows, int numCols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= numCols || y >= numRows) return;

    int idx = y * numCols + x;
    float mag = magnitude[idx];
    // convertir angulo a grados y quedarnos en [0, 180]
    float angle = direction[idx] * 180.0f / 3.14159f;
    if (angle < 0) angle += 180.0f;

    float n1 = 0, n2 = 0;

    // segun la direccion del gradiente, comparamos con los vecinos correspondientes
    if ((angle < 22.5f) || (angle >= 157.5f)) {         // horizontal
        if (x > 0)             n1 = magnitude[idx - 1];
        if (x < numCols - 1)   n2 = magnitude[idx + 1];
    }
    else if (angle < 67.5f) {                            // diagonal
        if (x > 0 && y > 0)                             n1 = magnitude[(y - 1) * numCols + (x - 1)];
        if (x < numCols - 1 && y < numRows - 1)         n2 = magnitude[(y + 1) * numCols + (x + 1)];
    }
    else if (angle < 112.5f) {                           // vertical
        if (y > 0)             n1 = magnitude[(y - 1) * numCols + x];
        if (y < numRows - 1)   n2 = magnitude[(y + 1) * numCols + x];
    }
    else {                                               // otra diagonal
        if (x < numCols - 1 && y > 0)           n1 = magnitude[(y - 1) * numCols + (x + 1)];
        if (x > 0 && y < numRows - 1)           n2 = magnitude[(y + 1) * numCols + (x - 1)];
    }

    // solo sobrevive si es el maximo entre sus dos vecinos
    output[idx] = (mag >= n1 && mag >= n2) ? mag : 0.0f;
}

// reduccion paralela para encontrar el valor maximo de un array
__global__
void find_max(const float* input, float* blockMaxes, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // cada thread carga su valor en shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // reduccion por mitades hasta quedar con el maximo del bloque
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    // el thread 0 escribe el maximo del bloque
    if (tid == 0) blockMaxes[blockIdx.x] = sdata[0];
}

// clasifica cada pixel
__global__
void double_threshold(const float* input, unsigned char* output,
    int numRows, int numCols, float highThresh, float lowThresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= numCols || y >= numRows) return;

    int idx = y * numCols + x;
    float val = input[idx];

    if (val >= highThresh)       output[idx] = 255; // borde fuerte
    else if (val >= lowThresh)   output[idx] = 128; // borde debil
    else                         output[idx] = 0;   // no es borde
}

// convierte imagen en escala de grises a RGBA para poder guardarla
__global__
void gray_to_rgba(const unsigned char* gray, uchar4* outputRGBA,
    int numRows, int numCols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= numCols || y >= numRows) return;

    int idx = y * numCols + x;
    unsigned char v = gray[idx];
    // mismo valor en R, G y B para escala de grises
    outputRGBA[idx] = make_uchar4(v, v, v, 255);
}

// convierte pixels debiles en fuertes si tienen algun vecino fuerte
__global__
void hysteresis(unsigned char* edges, int numRows, int numCols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= numCols || y >= numRows) return;

    int idx = y * numCols + x;
    if (edges[idx] != 128) return; // solo nos interesa los debiles

    // miramos los 8 vecinos
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < numCols && ny >= 0 && ny < numRows)
                if (edges[ny * numCols + nx] == 255) {
                    edges[idx] = 255; // promovemos a fuerte
                    return;
                }
        }
    }
    edges[idx] = 0; // ningun vecino fuerte, lo eliminamos
}

// convolucion sobre canal float (usado en Canny con filtros intermedios)
__global__
void convolution_float(const float* inputChannel, float* outputChannel,
    int numRows, int numCols,
    const float* filter, int filterWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= numCols || y >= numRows) return;

    int halfWidth = filterWidth / 2;
    float result = 0.0f;

    for (int fy = -halfWidth; fy <= halfWidth; fy++) {
        for (int fx = -halfWidth; fx <= halfWidth; fx++) {
            int nx = x + fx, ny = y + fy;
            float val = 0.0f;
            //si el vecino esta fuera, vale 0
            if (nx >= 0 && nx < numCols && ny >= 0 && ny < numRows)
                val = inputChannel[ny * numCols + nx];
            result += filter[(fy + halfWidth) * filterWidth + (fx + halfWidth)] * val;
        }
    }
    outputChannel[y * numCols + x] = result;
}

// convolucion sobre canal uchar (box filter normal)
__global__
void convolution(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   if (x >= numCols || y >= numRows) return;

   float result = 0.0f;
   int halfWidth = filterWidth / 2;

   for (int fy = -halfWidth; fy <= halfWidth; fy++) {
       for (int fx = -halfWidth; fx <= halfWidth; fx++) {
           int nx = x + fx;
           int ny = y + fy;

           float pixelValue = 0.0f;
           //vecinos fuera de la imagen valen 0
           if (nx >= 0 && nx < numCols && ny >= 0 && ny < numRows) {
               pixelValue = inputChannel[ny * numCols + nx];
           }

           int filterIdx = (fy + halfWidth) * filterWidth + (fx + halfWidth);
           // usar memoria de constantes o global segun el define activo
           #ifdef _CONSTANT_MEMORY
                      result += d_filter_const[filterIdx] * pixelValue;
           #else
                      result += filter[filterIdx] * pixelValue;
           #endif
       }
   }

   // clamp para que el resultado quede en 0 y 255
   result = min(255.0f, max(0.0f, result));
   outputChannel[y * numCols + x] = (unsigned char)result;
}

//carga el tile + halo en shared memory
__global__
void convolution_shared(const unsigned char* const inputChannel,
    unsigned char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{
    int halfWidth = filterWidth / 2;
    // ancho del tile en shared memory
    int sharedWidth = TILE_WIDTH + 2 * halfWidth;
    int sharedHeight = TILE_HEIGHT + 2 * halfWidth;

    // shared memory con tamaño maximo para filtro 9x9
    __shared__ float sharedTile[(TILE_HEIGHT + 2 * MAX_HALF_WIDTH) * (TILE_WIDTH + 2 * MAX_HALF_WIDTH)];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

    int threadIdx1D = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int totalThreads = TILE_WIDTH * TILE_HEIGHT;
    int totalPixels = sharedWidth * sharedHeight;

    //cada thread carga uno o varios pixels del tile + halo
    for (int i = threadIdx1D; i < totalPixels; i += totalThreads) {
        int sx = i % sharedWidth;
        int sy = i / sharedWidth;

        int imgX = blockIdx.x * TILE_WIDTH + sx - halfWidth;
        int imgY = blockIdx.y * TILE_HEIGHT + sy - halfWidth;

        if (imgX >= 0 && imgX < numCols && imgY >= 0 && imgY < numRows)
            sharedTile[i] = inputChannel[imgY * numCols + imgX];
        else
            sharedTile[i] = 0.0f; // halo fuera de imagen = 0
    }

    // esperamos a que todos hayan cargado su parte
    __syncthreads();

    if (x >= numCols || y >= numRows) return;

    // calculo de convolucion leyendo de shared memory en vez de global
    float result = 0.0f;
    for (int fy = 0; fy < filterWidth; fy++) {
        for (int fx = 0; fx < filterWidth; fx++) {
            int sx = threadIdx.x + fx;
            int sy = threadIdx.y + fy;
#ifdef _CONSTANT_MEMORY
            result += d_filter_const[fy * filterWidth + fx] * sharedTile[sy * sharedWidth + sx];
#else
            result += filter[fy * filterWidth + fx] * sharedTile[sy * sharedWidth + sx];
#endif
        }
    }

    result = min(255.0f, max(0.0f, result));
    outputChannel[y * numCols + x] = (unsigned char)result;
}

// separa imagen RGBA en 3 canales independientes R, G, B
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

    uchar4 pixel = inputImageRGBA[thread_1D_pos];

    // x=R, y=G, z=B
    redChannel[thread_1D_pos]   = pixel.x;
    greenChannel[thread_1D_pos] = pixel.y;
    blueChannel[thread_1D_pos]  = pixel.z;
}

// junta los 3 canales R, G, B en una imagen RGBA
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  // alpha = 255 (sin transparencia)
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);
  outputImageRGBA[thread_1D_pos] = outputPixel;
}

// canales separados en GPU (globales para no tener que pasarlos entre funciones)
unsigned char *d_red, *d_green, *d_blue;
float* d_filter;

// reserva memoria en GPU para los 3 canales de color
void allocateMemoryGPU(const size_t numRowsImage, const size_t numColsImage)
{
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
}

// sube el filtro a GPU (a memoria de constantes o global segun el define)
void allocateFilterAndCopyToGPU(const float *h_filter, const size_t filterWidth, float **d_filter)
{
    #ifdef _CONSTANT_MEMORY
        // memoria de constantes: no hay malloc, se copia directamente al simbolo
        cudaMemcpyToSymbol(d_filter_const, h_filter, sizeof(float) * filterWidth * filterWidth);
    #else
        checkCudaErrors(cudaMalloc(d_filter, sizeof(float) * filterWidth * filterWidth));
        checkCudaErrors(cudaMemcpy(*d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
    #endif
}

// libera la memoria reservada en GPU
void cleanupGPU() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
    #ifndef _CONSTANT_MEMORY
      checkCudaErrors(cudaFree(d_filter));
    #endif
}

// crea el filtro en CPU segun el id pasado por argumento
void create_filter(float **h_filter, int *filterWidth, int id_filter){

  switch ( id_filter )
  {
    case 0: // Gaussiano 9x9
    {
      FILTERSIZE = 9;
      *filterWidth = 9;
      int KernelWidth = 9;
      *h_filter = new float[KernelWidth * KernelWidth];

      const float KernelSigma = 2.;
      float filterSum = 0.f;

      // calcular valores gaussianos
      for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
        for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
          float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
          (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
          filterSum += filterValue;
        }
      }

      // normalizar para que la suma sea 1
      float normalizationFactor = 1.f / filterSum;
      for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
        for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
          (*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
        }
      }
    }
    break;

    case 1: // Laplaciano 5x5
    {
      FILTERSIZE = 5;
      *filterWidth = 5;
      *h_filter = new float[25];

      (*h_filter)[0] = 0;    (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
      (*h_filter)[5] = 0;    (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
      (*h_filter)[10] = -1.; (*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
      (*h_filter)[15] = 0;   (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
      (*h_filter)[20] = 0;   (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;
    }
    break;

    case 2: // Sharpen 3x3
    {
        FILTERSIZE = 3;
        *filterWidth = 3;
        *h_filter = new float[3 * 3];
        (*h_filter)[0] =  0; (*h_filter)[1] = -1; (*h_filter)[2] =  0;
        (*h_filter)[3] = -1; (*h_filter)[4] =  5; (*h_filter)[5] = -1;
        (*h_filter)[6] =  0; (*h_filter)[7] = -1; (*h_filter)[8] =  0;
    }
    break;

    case 3: // Sobel horizontal 3x3
    {
        FILTERSIZE = 3;
        *filterWidth = 3;
        *h_filter = new float[3 * 3];
        (*h_filter)[0] = -1; (*h_filter)[1] = 0; (*h_filter)[2] = 1;
        (*h_filter)[3] = -2; (*h_filter)[4] = 0; (*h_filter)[5] = 2;
        (*h_filter)[6] = -1; (*h_filter)[7] = 0; (*h_filter)[8] = 1;
    }
    break;

    case 4: // Sobel vertical 3x3
    {
        FILTERSIZE = 3;
        *filterWidth = 3;
        *h_filter = new float[3 * 3];
        (*h_filter)[0] = -1; (*h_filter)[1] = -2; (*h_filter)[2] = -1;
        (*h_filter)[3] =  0; (*h_filter)[4] =  0; (*h_filter)[5] =  0;
        (*h_filter)[6] =  1; (*h_filter)[7] =  2; (*h_filter)[8] =  1;
    }
    break;

    case 5: // Deteccion de bordes
    {
        FILTERSIZE = 3;
        *filterWidth = 3;
        *h_filter = new float[3 * 3];
        (*h_filter)[0] = -1; (*h_filter)[1] = -1; (*h_filter)[2] = -1;
        (*h_filter)[3] = -1; (*h_filter)[4] =  8; (*h_filter)[5] = -1;
        (*h_filter)[6] = -1; (*h_filter)[7] = -1; (*h_filter)[8] = -1;
    }
    break;

    default:
      printf("Filtro no definido\n");
      exit(1);
  }
}


void box_filter(uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
    unsigned char* d_redFiltered,
    unsigned char* d_greenFiltered,
    unsigned char* d_blueFiltered,
    int id_filter)
{
  float *h_filter;
  int filterWidth;

  // para box filter: reservar canales, crear filtro y subirlo a GPU
#ifndef _CANNY_EDGE
  allocateMemoryGPU(numRows, numCols);
  create_filter(&h_filter, &filterWidth, id_filter);
  allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter);
#endif

  // tamaño de bloque fijo 32x32, grid se adapta al tamaño de la imagen
  const dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
  const dim3 gridSize((numCols + TILE_WIDTH - 1) / TILE_WIDTH,
      (numRows + TILE_HEIGHT - 1) / TILE_HEIGHT);

#ifdef _CANNY_EDGE

  // buffers intermedios para el pipeline de Canny
  float* d_gray, * d_blurred, * d_magnitude, * d_direction, * d_suppressed;
  unsigned char* d_edges;
  size_t numPixels = numRows * numCols;
  cudaMalloc(&d_gray,       sizeof(float) * numPixels);
  cudaMalloc(&d_blurred,    sizeof(float) * numPixels);
  cudaMalloc(&d_magnitude,  sizeof(float) * numPixels);
  cudaMalloc(&d_direction,  sizeof(float) * numPixels);
  cudaMalloc(&d_suppressed, sizeof(float) * numPixels);
  cudaMalloc(&d_edges,      sizeof(unsigned char) * numPixels);

  // paso a escala de grises
  grayscale<<<gridSize, blockSize>>>(d_inputImageRGBA, d_gray, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //gaussian blur para eliminar ruido
  float* h_gauss, * d_gauss; int gWidth;
  create_filter(&h_gauss, &gWidth, 0);
  cudaMalloc(&d_gauss, sizeof(float) * gWidth * gWidth);
  cudaMemcpy(d_gauss, h_gauss, sizeof(float) * gWidth * gWidth, cudaMemcpyHostToDevice);
  convolution_float<<<gridSize, blockSize>>>(d_gray, d_blurred, numRows, numCols, d_gauss, gWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  cudaFree(d_gauss);
  delete[] h_gauss;

  //calculo gradiente con Sobel
  gradient_magnitude_direction<<<gridSize, blockSize>>>(d_blurred, d_magnitude, d_direction, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // suprimir no-maximos para afinar bordes
  non_max_suppression<<<gridSize, blockSize>>>(d_magnitude, d_direction, d_suppressed, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //reduccion para encontrar el valor maximo (necesario para los umbrales)
  int blockSize1D = 256;
  int gridSize1D = (numPixels + blockSize1D - 1) / blockSize1D;
  float* d_blockMaxes;
  cudaMalloc(&d_blockMaxes, sizeof(float) * gridSize1D);
  find_max<<<gridSize1D, blockSize1D, blockSize1D * sizeof(float)>>>(d_suppressed, d_blockMaxes, numPixels);
  find_max<<<1, blockSize1D, blockSize1D * sizeof(float)>>>(d_blockMaxes, d_blockMaxes, gridSize1D);
  float h_max;
  cudaMemcpy(&h_max, d_blockMaxes, sizeof(float), cudaMemcpyDeviceToHost);

  //clasificar pixels como fuertes, debiles o irrelevantes
  double_threshold<<<gridSize, blockSize>>>(d_suppressed, d_edges, numRows, numCols,
      0.2f * h_max, 0.1f * h_max);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //hysteresis - promover debiles con vecinos fuertes
  for (int i = 0; i < 5; i++) {
      hysteresis<<<gridSize, blockSize>>>(d_edges, numRows, numCols);
      cudaDeviceSynchronize();
  }

  //convertir resultado a RGBA para guardar la imagen
  gray_to_rgba<<<gridSize, blockSize>>>(d_edges, d_outputImageRGBA, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // liberar todos los buffers intermedios
  cudaFree(d_gray);      cudaFree(d_blurred);
  cudaFree(d_magnitude); cudaFree(d_direction);
  cudaFree(d_suppressed); cudaFree(d_edges);
  cudaFree(d_blockMaxes);

#else

  // separar imagen en canales R, G, B
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols,
      d_red, d_green, d_blue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // aplicar convolucion a cada canal por separado
#ifdef _SHARED_MEMORY
  convolution_shared<<<gridSize, blockSize>>>(d_red,   d_redFiltered,   numRows, numCols, d_filter, filterWidth);
  convolution_shared<<<gridSize, blockSize>>>(d_green, d_greenFiltered, numRows, numCols, d_filter, filterWidth);
  convolution_shared<<<gridSize, blockSize>>>(d_blue,  d_blueFiltered,  numRows, numCols, d_filter, filterWidth);
#else
  convolution<<<gridSize, blockSize>>>(d_red,   d_redFiltered,   numRows, numCols, d_filter, filterWidth);
  convolution<<<gridSize, blockSize>>>(d_green, d_greenFiltered, numRows, numCols, d_filter, filterWidth);
  convolution<<<gridSize, blockSize>>>(d_blue,  d_blueFiltered,  numRows, numCols, d_filter, filterWidth);
#endif
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // juntar los 3 canales filtrados en la imagen de salida
  recombineChannels<<<gridSize, blockSize>>>(d_redFiltered, d_greenFiltered, d_blueFiltered,
      d_outputImageRGBA, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

#endif

}

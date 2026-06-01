/*
Author: Cao Thanh Tung and Zheng Jiaqi
Date: 21/01/2010, 20/08/2019

File Name: pba2D.h

===============================================================================

Copyright (c) 2019, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef __CUDA_H__
#define __CUDA_H__

// Initialize CUDA and allocate memory
// textureSize is 2^k with k >= 6
extern "C" void pba2DInitialization(int textureSize, int phase1Band); 

// Deallocate memory in GPU
extern "C" void pba2DDeinitialization(); 

// Compute 2D Voronoi diagram
// Input: a 2D texture. Each pixel is represented as two "short" integer. 
//    For each site at (x, y), the pixel at coordinate (x, y) should contain 
//    the pair (x, y). Pixels that are not sites should contain the pair (MARKER, MARKER)
// Output: 2 2D texture. Each pixel is represented as two "short" integer 
//    refering to its nearest site. 
// See original paper for the effect of the three parameters: 
//     phase1Band, phase2Band, phase3Band
// Parameters must divide textureSize
extern "C" void pba2DVoronoiDiagram(short *input, short *output, int phase1Band,
                                    int phase2Band, int phase3Band);

// MARKER is used to mark blank pixels in the texture.
// Any uncolored pixels will have x = MARKER.
// Input texture should have x = MARKER for all pixels other than sites
#define MARKER      -32768

// ===========================================================================
// Lower-level / device-resident API.
// (Added for the gpu-edt-2d-benchmark packaging: declarations only — the
//  implementations and all kernels in pba2DKernel.h are the original NUS code,
//  unchanged. This just publishes the public symbols so callers don't have to
//  redeclare them.)
// ===========================================================================
#include <vector_types.h>   // short2

// Upload a host site map into the device input buffer (H2D). Destructive:
// pba2DCompute overwrites the buffer, so call this before each Compute.
void pba2DInitializeInput(short *input);

// Run the three PBA phases on the current device input buffer (no H2D/D2H).
// Bands must divide textureSize. Result lands in the output texture.
void pba2DCompute(int phase1Band, int phase2Band, int phase3Band);

// Direct device pointers to the input/output short2 textures, for fully
// device-resident use (write sites into Input; read nearest-site from Output
// after pba2DCompute).
extern "C" short2 *pba2DInputDevice();
extern "C" short2 *pba2DOutputDevice();

#endif

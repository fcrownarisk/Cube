# Cube4096 System Documentation

## Overview
A pure C implementation of a 4096-cell cube (16×16×16) with:
1. **1+1=2 Mathematical Foundation** - All operations based on this principle
2. **4-Color System** - Visual representation using 4 distinct colors
3. **XYZNNN Coordinate System** - Enhanced 3D coordinates with additional parameters

## Features

### Core Components:
1. **Cube Structure**: 4096 cells with XYZ coordinates and NNN parameters
2. **Mathematical Engine**: Operations based on the 1+1=2 theorem
3. **Visualization**: 2D slices, 3D projections, color mapping
4. **Pattern Generation**: Wave functions, fractal patterns, mathematical interference

### 4-Color System:
- RED (0): Low values (0-0.25 normalized)
- GREEN (1): Medium-low values (0.25-0.5)
- BLUE (2): Medium-high values (0.5-0.75)
- YELLOW (3): High values (0.75-1.0)

### XYZNNN System:
- X, Y, Z: Standard 3D coordinates (0-15)
- N1, N2, N3: Additional parameters for mathematical operations
- Pattern: Mathematical pattern identifier

## Building the System

### Requirements:
- GCC or compatible C compiler
- Math library (-lm)
- Terminal with ANSI color support

### Build Commands:
```bash
# Standard build
make

# Debug build
make debug

# Release build (optimized)
make release

# Run the program
make run

# Clean build files
make clean
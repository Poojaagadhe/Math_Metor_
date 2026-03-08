# Linear Algebra Basics

## Matrices

A matrix is a rectangular array of numbers:
```
A = [a₁₁  a₁₂  a₁₃]
    [a₂₁  a₂₂  a₂₃]
```

**Matrix Dimensions**: m × n (m rows, n columns)

## Matrix Operations

**Addition**: Add corresponding elements (matrices must have same dimensions)
```
[a b] + [e f] = [a+e  b+f]
[c d]   [g h]   [c+g  d+h]
```

**Scalar Multiplication**: Multiply each element by scalar k
```
k·[a b] = [ka  kb]
  [c d]   [kc  kd]
```

**Matrix Multiplication**: (AB)ᵢⱼ = Σ AᵢₖBₖⱼ
- Number of columns in A must equal number of rows in B
- Result has dimensions: (rows of A) × (columns of B)

## Determinants

**2×2 Matrix**:
```
det([a b]) = ad - bc
    [c d]
```

**3×3 Matrix** (using cofactor expansion):
```
det([a b c]) = a·det([e f]) - b·det([d f]) + c·det([d e])
    [d e f]         [h i]         [g i]         [g h]
    [g h i]
```

**Properties**:
- det(AB) = det(A)·det(B)
- det(Aᵀ) = det(A)
- If det(A) = 0, matrix is singular (not invertible)

## Matrix Inverse

For 2×2 matrix:
```
A⁻¹ = (1/det(A))·[d  -b]
                 [-c  a]
```

Where A = [a b]
          [c d]

**Properties**:
- AA⁻¹ = A⁻¹A = I (identity matrix)
- (AB)⁻¹ = B⁻¹A⁻¹
- (Aᵀ)⁻¹ = (A⁻¹)ᵀ

## Vectors

**Dot Product**:
```
u·v = u₁v₁ + u₂v₂ + u₃v₃
```

**Properties**:
- u·v = |u||v|cos(θ)
- If u·v = 0, vectors are orthogonal

**Cross Product** (3D vectors):
```
u × v = [u₂v₃ - u₃v₂]
        [u₃v₁ - u₁v₃]
        [u₁v₂ - u₂v₁]
```

**Magnitude**:
```
|v| = √(v₁² + v₂² + v₃²)
```

## Linear Transformations

A transformation T: Rⁿ → Rᵐ is linear if:
1. T(u + v) = T(u) + T(v)
2. T(cu) = cT(u)

Every linear transformation can be represented by a matrix.

## Common Mistakes

1. **Matrix multiplication**: Trying to multiply incompatible dimensions
2. **Non-commutativity**: Assuming AB = BA (generally false for matrices)
3. **Determinant**: Sign errors in cofactor expansion
4. **Inverse**: Forgetting to check if det(A) ≠ 0
5. **Dot vs Cross product**: Confusing the two operations

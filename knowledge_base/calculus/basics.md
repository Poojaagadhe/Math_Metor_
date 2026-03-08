# Calculus Basics

## Limits

The limit of f(x) as x approaches a:
```
lim(x→a) f(x) = L
```

**Limit Laws**:
- Sum: lim(f + g) = lim f + lim g
- Product: lim(fg) = (lim f)(lim g)
- Quotient: lim(f/g) = (lim f)/(lim g), provided lim g ≠ 0

**Important Limits**:
```
lim(x→0) (sin x)/x = 1
lim(x→∞) (1 + 1/x)ˣ = e
```

**L'Hôpital's Rule**: For indeterminate forms 0/0 or ∞/∞:
```
lim(x→a) f(x)/g(x) = lim(x→a) f'(x)/g'(x)
```

## Derivatives

The derivative of f(x):
```
f'(x) = lim(h→0) [f(x+h) - f(x)]/h
```

**Basic Derivative Rules**:
- Constant: d/dx(c) = 0
- Power rule: d/dx(xⁿ) = nxⁿ⁻¹
- Constant multiple: d/dx(cf) = c·f'
- Sum: d/dx(f + g) = f' + g'

**Product Rule**:
```
d/dx(fg) = f'g + fg'
```

**Quotient Rule**:
```
d/dx(f/g) = (f'g - fg')/g²
```

**Chain Rule**:
```
d/dx(f(g(x))) = f'(g(x))·g'(x)
```

**Common Derivatives**:
- d/dx(sin x) = cos x
- d/dx(cos x) = -sin x
- d/dx(eˣ) = eˣ
- d/dx(ln x) = 1/x

## Optimization

To find maximum/minimum of f(x):
1. Find critical points: f'(x) = 0
2. Test using second derivative:
   - If f''(x) > 0: local minimum
   - If f''(x) < 0: local maximum
   - If f''(x) = 0: inconclusive (use first derivative test)

**Steps for Applied Optimization**:
1. Identify the quantity to optimize
2. Express it as a function of one variable
3. Find the domain
4. Find critical points
5. Test endpoints and critical points
6. Verify the answer makes sense

## Common Mistakes

1. **Chain rule**: Forgetting to multiply by the inner derivative
2. **Product/Quotient rules**: Mixing up the formulas
3. **Critical points**: Forgetting to check where derivative doesn't exist
4. **Optimization**: Not checking endpoints of the domain
5. **L'Hôpital's Rule**: Applying when form is not indeterminate

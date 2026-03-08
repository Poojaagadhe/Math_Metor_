# Probability Fundamentals

## Basic Probability

**Probability of Event A**:
```
P(A) = (Number of favorable outcomes) / (Total number of outcomes)
```

**Properties**:
- 0 ≤ P(A) ≤ 1
- P(certain event) = 1
- P(impossible event) = 0
- P(not A) = 1 - P(A)

## Conditional Probability

Probability of A given B has occurred:
```
P(A|B) = P(A ∩ B) / P(B)
```

**Bayes' Theorem**:
```
P(A|B) = [P(B|A) · P(A)] / P(B)
```

## Independent Events

Events A and B are independent if:
```
P(A ∩ B) = P(A) · P(B)
```

Or equivalently:
```
P(A|B) = P(A)
```

## Permutations and Combinations

**Permutations** (order matters):
```
P(n, r) = n! / (n-r)!
```

Number of ways to arrange r items from n items.

**Combinations** (order doesn't matter):
```
C(n, r) = n! / [r!(n-r)!]
```

Number of ways to choose r items from n items.

**Notation**: C(n, r) is also written as ⁿCᵣ or (n choose r)

## Addition and Multiplication Rules

**Addition Rule** (for mutually exclusive events):
```
P(A or B) = P(A) + P(B)
```

**General Addition Rule**:
```
P(A or B) = P(A) + P(B) - P(A and B)
```

**Multiplication Rule** (for independent events):
```
P(A and B) = P(A) · P(B)
```

**General Multiplication Rule**:
```
P(A and B) = P(A) · P(B|A)
```

## Common Distributions

**Binomial Distribution**: n independent trials, each with probability p of success
```
P(X = k) = C(n, k) · pᵏ · (1-p)ⁿ⁻ᵏ
```

## Common Mistakes

1. **Conditional probability**: Confusing P(A|B) with P(B|A)
2. **Independence**: Assuming events are independent without verification
3. **Addition rule**: Forgetting to subtract P(A ∩ B) for non-exclusive events
4. **Permutations vs Combinations**: Using permutations when order doesn't matter
5. **Bayes' theorem**: Incorrect application of prior and posterior probabilities

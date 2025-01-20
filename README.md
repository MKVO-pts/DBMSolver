# DBMSolver
Decompose High order Data-Based Mechanistic models(DBM) models into a combination of 1st order models using rust. It returns the mathematical possible conformations with the 1st Order models characteristics

## SISO
*Single input, single output

Because of knowledge limitations, it is only done for Transfer Functions from a SISO situation. 
In theory it should be possible to convert a State Space Model to a Transfer Function ("MIMO" to a SISO) and then use this functions, but the code will not be writien considering those cases (for now at least)

## Methods
1. Reducing the model order by polynomial division (only when m>n)
2. Decomposition into first order models
3. Removing unstable configurations (|a|>1)
4. Interpreting the model structure mechanistically (create Tree)

### Input
The input should be two lists with the Exo and Endo parameters (A,B), from the higher to lower order (or lower to higher?). 
``[5,3,3,6,2] = 5x⁴+3x³+3x²+6x+2``

``[7,3,0,3,0] = 7x⁴+3x³+0x²+2x+0 = 7x⁴+3x³+2x``
### Output
The output should be a Phylogenetic-like Tree were each node represents a 1stOrder model. In a form of vector, where each nested vector represents a division.
The first order vectors, the First part is the "A" and the Second the "B"
```bash
['p' = ['s' = [[3,5],[2,1],[4,2],'f' = [[8,3],[1,3]]]]

//OR

[
  'p' = [
    's' = [
      [3,5],
      [2,1],
      [4,2]
      ],
    'f' = [
      [8,3],
      [1,3]
      ]
    ]
]
```
"s" = serial;
"p" = parallel;
"f" = feedback


NOTE: Maybe first do it using Python, then convert it to RUST

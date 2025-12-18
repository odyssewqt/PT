## scheme-v1: cross-attention 64 local & 1  global 
## scheme-v2: cross-attention 64 local & 64 global

## pt-v1: cross-attention 
  - 最底层：64 local & 64 global
  - 其他层：64 local & 1  global
  
### True
![True](./results/fluid-true.gif)

### Pre
![Pre](./results/fluid-pre.gif)
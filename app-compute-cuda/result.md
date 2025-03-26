Example applications showing core API features, common patterns and best
practices.

# benchmark reuslt 
Date: 26-3-2025


| Type   | Category      | Average time(us) | Standard deviation(us) |
| ------ | ------------- | ------------ | ------------------ |
| Server | device alloc  | 575.4        | 46.8               |
| Client | device alloc  | 658.6        | 46.9               |
| Server | rpc           | 8.1          | 0.3                |
| Client | rpc           | 35.6         | 2.8                |
| Client | host alloc    | 1398.8       | 83.3               |
| Client | copy*2        | 2155.7       | 111.1              |
| Server | load module   | 87.9         | 2.9                |
| Client | load module   | 167.0        | 5.3                |
| Server | get function  | 9.6          | 0.5                |
| Client | get function  | 79.7         | 2.6                |
| Server | launch kernel | 17.3         | 0.5                |
| Client | launch kernel | 36.3         | 1.3                |
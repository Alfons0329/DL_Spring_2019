# Some good combinations for HW1 problem1
* Batch size and learning rate to choose
|Batch size|eta(learning rate)|
|----------|------------------|
|8|1e-5|
|4|1e-5|
|50|1e-5|
|25|1e-06|
|40|1e-06|


* 所以整體流程
```
我先構件出一個 w1 w2 w3..等等 隨機的
然後我在用 SGD找優話，過程中因為 backpropogation 從 w3 逆推回去就會用連鎖律帶出w3 2 1的關係
最後 隨著SGD的優話就會讓錯誤率下降
但是要下降到什麼程度，什麼時候該停，下降的好不好，就是要看 
會影響 SGD 進行的參數，這樣
```

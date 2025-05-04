Ref: https://docs.google.com/document/d/1ztFALf58Ge-MO2Lk1O48mk-wXurR6EvEHrPICfWbB78/edit?tab=t.0


# DRONE VISION PROJECT

# V1

As the dataset is not too large, I tried first with the same advanced 3 layer CNN that was provided to us in practice with 1 tiny change that i dont predict classes as this is just 1 class dataset. I trained it further for few hunder epocs and it automatically got really good results and highest result on public leaderboad. 

# V2

To improve results i tried just adding 1 more layer and training for a day. Improved 5 % 


# V3

Then i added in total 7 layers and tried again and got 3 % percent improvemnt. 

I dont think I could get anything more from adding extra layers withoud residual connections but it seems that the improvent line on IOUs still goes further. What happens if i train for 24 hours further? When will I converge? 
1. 38 hours of training with 7 layers bought 2 % gains 
2. +24 hours of training did not bring any gainds


# V4 

Trying out 3 layers with 24 hours of training to see if this can achieve comparable results. Got 92 % IOU, which is not best but a littlebit worse. 2900 epochs


# CONCLUSION

- Probably the main advancment of my NN was that i removed class prediction within this 1 class dataset. This allowed the training to focus on finding right size and location and confidence of object being there
- Training for a day got the best result 
- V3 model with 7 layers after 2900 epochs got best result of IOU of 93.2. Intuitively its not that much better than 3 layer V4 network agter 2900 epochs that got 92.8 IOU.
- It seems tiny very specific network can get relatively good without crasy amount of training and data. [Discussion about this](https://chatgpt.com/share/68172f52-2f90-8008-b860-5e8eff4e8b53)





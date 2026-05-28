logging
- number of distinct clusters -- need to add this to verl. also: response length, critic mean score. 
- number of prompts unlocked (unique number of prompts solved, throughout training)
- critic mean score/critic mean reward
- mean response length

try moving to verl
- implement both arms on verl
- this ensures that there's no issue with our coding. 
- "And then to satisfy coding component for the project you could implement set rl with minority voting objective on verl." 
- verl gives us the answer extraction. 

changes we should make
- up to batch size 128, splitting across GPUs (verl does this for us)
- we should try clustering by CoT... answer clustering isn't great as that sort of just leads to reward hacking. 
- answer extraction: just use whatever is built in to verl. MathReward to extract the answer. 
- use 4B if we can fit it, which we should be able to fit it. 
- llm judge asynchronously (speeds up training a lot) (1 judge, just batch the 64 calls through it. or do 32 if it doesn't fit. so you only need to host the judge once, but you do the 32 or 64 calls in parallel). Used semaphore from asyncio to "do 128 at the same time" -- i think this is more for remote calling the API. 
- figure out how to make the API for the judge. they hosted the GPUs locally so they just made an API. We need to figure out how to do that with Modal -- don't know if VeRL will handle this part for us. 

should we use filtered dataset or not filtered?
- can keep using the filtered dataset

online or offline training?
- online

what metrics should we monitor during training, what should we expect to see, and what is BAD to see? should we do mid-run evals to see that training is progressing healthily?
- yes do mid-run eval ckpt stuff
- we should see that the critic mean score is increasing (average value function). critic mean reward. we should see that training reward is increasing. score basically same as reward in this context. (verl sometimes uses mean score and sometimes uses mean reward)
- also look at mean response length (can tell u whether it's outputting gibberish, or look at # of rollouts put in degenerate cluster), number of clusters. 

what should we do if we still dont see good results during training? 
- if it fails, just do an analysis... see why the minority voting thing fails. it has not been tried before. we're still the first ones to do it. find insights, do analysis, on why we're seeing the results we're seeing with minority voting. 

DO POLY EPO CHAIN OF THOUGHT as 3rd run to see what is going on. could lead to cool analysis. 
- VeRL "trainer.log_val_generations=10"
- can also make our own eval scripts. 
- NEED TO DO AIME 25 MID-RUN VALIDATION, FOR EXAMPLE. OR 2K POLARIS DATASET. 
- actually, use maxRL- they implement their own reward function as a good example, plus verl is now contaminated bc of many contributors… their fork is clean

1 epoch vs 2 epochs?
- good to be able to train for more. sometimes with 1 epoch u dont see the results you're expecting. 

answer extraction method?
- move to mathreward + \boxed{}

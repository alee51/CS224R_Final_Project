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

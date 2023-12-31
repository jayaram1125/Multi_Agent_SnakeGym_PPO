Multi Snake Gym implemented using Box2d and Envpool libraries

PPO(Proximal policy optimization) algorithm  with multiple actor critics used to train the snake agents. This project is modification and extension of  https://github.com/jayaram1125/Single_Agent_SnakeGym_PPO



Training duration : 102 hrs for 20000 updates using 1 RTXA6000 48 GB GPU and 10 CPU machine for each strategy shown below
<br/>
<br/>


                                              COOPERATIVE STRATEGY
<p>
    <img width="1600" height="400" src="https://github.com/jayaram1125/Multi_Agent_SnakeGym_PPO/blob/main/Cooperative_Output/Cooperative.png">
    <img title="Green Snake Completes Game" width="400" height="400" src="https://github.com/jayaram1125/Multi_Agent_SnakeGym_PPO/blob/main/Cooperative_Output/TrainOutput/Episode_5228_GreenSnakeWin_GIF_0.25x.gif">
    <img title="Blue Snake Completes Game" width="400" height="400" src="https://github.com/jayaram1125/Multi_Agent_SnakeGym_PPO/blob/main/Cooperative_Output/TestOutput/Test_Episode_2_BlueSnakeWinGIF_0.25x.gif" hspace="50">
</p>

<br/>
<br/>

                                               COMPETITIVE STRATEGY

<p>
    <img width="1600" height="400" src="https://github.com/jayaram1125/Multi_Agent_SnakeGym_PPO/blob/main/Competitive_Output/Competitive.png">
    <img title="Green Snake Completes Game" width="400" height="400" src="https://github.com/jayaram1125/Multi_Agent_SnakeGym_PPO/blob/main/Competitive_Output/TrainOutput/Episode_5018_GreenSnakeWin_0.0625x_GIF.gif">
    <img title="Blue Snake Completes Game" width="400" height="400" src="https://github.com/jayaram1125/Multi_Agent_SnakeGym_PPO/blob/main/Competitive_Output/Test_output/Test_Episode_4_BlueSnakeWin_GIF_0.25x.gif" hspace="50">
</p>

<br/>
<br/>

Note:Except the files mpi_pytorch.py, mpi_tools.py,ppo.py, video_recorder.py, multisnake_test.py , all other files have to be placed in the path envpool/envpool/box2d folder to build and install the environment <br/>

1.To build envpool : make bazel-build   in the path ~/envpool   <br/>
2.To install envpool: pip install /home/jayaram/SnakeGame/envpool/dist/envpool-0.6.7-cp39-cp39-linux_x86_64.whl <br/>
3.To train and test model: AVAI_DEVICES=0 RCALL_NUM_GPU=1 mpiexec -np 1 python3 -m multi_agent_ppo <br/>
4.GIFS are captured in slow motion to show the interaction between snake agents more clearly <br/>

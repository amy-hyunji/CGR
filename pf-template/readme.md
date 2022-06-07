### Related Sites
* https://github.com/friendliai/periflow-cli
* https://github.com/friendliai/periflow-python-sdk
* example of pl: https://github.com/friendliai/periflow-python-sdk/tree/main/examples/pth-lightning

### Commands
```
% login
>> pf login
    * id: hyunji
    * pwd: hyunjilee

% datastore
>> pf datastore upload -n [name] -p [source path]

% run job
>> pf job run -f [config file] -d [src file] 

% checkpoint
>> pf checkpoint list

% see VM types
>> pf vm list

% plugin setting
* create credential
    >> pf credential create -h
    use id of created credential
* slack channel 
    * slack -> setting -> copy member ID
```

### Config
```
% The name of experiment  
experiment: 

% The name of job   
name: 

% The name of vm type
vm: 

% The number of GPU devices
num_devices: 

% Configure your job!  
job_setting:  
  type: custom  

  % Docker config  
  docker:  
    % Docker image you want to use in the job  
    image: friendliai/periflow:sdk  
    % Bash shell command to run the job  
    % NOTE: PeriFlow automatically sets the following environment variables for PyTorch DDP.  
    %   - MASTER_ADDR: Address of rank 0 node.  
    %   - WORLD_SIZE: The total number of GPUs participating in the task.  
    %   - NODE_RANK: Index of the current node. 
    %   - NPROC_PER_NODE: The number of processes in the current node.   
    command: >  
      cd /workspace/src && pip install -r requirements.txt && cd transformers && pip install -e . && cd ../ && python train.py --config config/periflow/gr/nq_toy_bi_p1-5_first_only.json  
  
  % Path to mount your workspace volume  
  workspace:  
    mount_path: /workspace  

% Checkpoint config  
checkpoint:  
  input: % to resume training, remove if unnecessary  
      id: % uuid of ckpt to resume   
      mount_path: % same path as it was saved   
  % Path to output checkpoint  
  output_checkpoint_dir: /workspace/ckpt  

data:  
   name: % name of datastore to load   
   mount_path: /workspace/dataset  

plugin:
   wandb: 
      credential_id: 
   slack:
      credential_id: 
      channel: 
```
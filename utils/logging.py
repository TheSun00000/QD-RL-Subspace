import neptune


    
def init_neptune(tags=[], mode='async'):

    run = neptune.init_run(
        project="nazimbendib1/QD-RL-Subspace",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGIxMzhlZS00MzhhLTQ0ZDktYTU2Yy0yZDk3MjE4MmU4MDgifQ==",
        tags=tags,
        mode=mode
    )
    
    return run
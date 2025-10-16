graph LR
    %% Core Stages
    A[Raw Data Sources] --> D[Data Preparation]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Finalized Model]
    G --> H[Predictions]
    
    %% Iteration Loops
    F --> |"Refine"| D
    F --> |"Retrain"| E

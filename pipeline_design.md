```mermaid
graph TD
    %% Raw Data Ingestion
    A[raw_data] --> A1[data_ingestion]
    A1 --> A2[(DuckDB - Raw Data)]
    A1 --> A3[(DuckDB - Labels)]
    
    %% Feature Processing
    A1 --> B[processed_features]
    B --> B1[(DuckDB - Features)]
    
    %% Feature Engineering Branch
    B --> C[feature_statistics]
    B --> D[predictions]
    
    %% Prediction Branch
    P --> F[current_model]
    F --> D[predictions]
    B --> D
    
    %% Monitoring Branch
    B1 --> H[updated_feature_statistics]
    O --> I[drift_alert]
    H --> O
    
    %% Model Training Branch (triggered by drift)
    I --> J[training_data]
    B1 --> J
    A3 --> J
    J --> K[trained_model]
    K --> L[model_validation]
    L --> M[model_registry_update]
    M --> F
    D --> G[prediction_logs]
    
    %% Performance Monitoring Branch
    D --> T[compute_performance]
    A3 --> T
    T --> U[(DuckDB - Performance)]
    
    %% Storage Assets
    G --> N[(DuckDB - Prediction Logs)]
    C --> O[(DuckDB - Feature Stats)]
    M --> P[(DuckDB - Model Registry)]
    
    %% Sensors and Jobs
    Q{drift_sensor} --> I
    R{scheduled_training} --> J
    S{prediction_job} --> A
    U --> V{performance_sensor}
    V --> I
    
    %% Styling
    classDef dataAsset fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef mlAsset fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef storageAsset fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef sensorAsset fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class A,A1,B,C,G,H,J,T dataAsset
    class D,I,F,K,L,M mlAsset
    class A2,A3,B1,N,O,P,U storageAsset
    class Q,R,S,V sensorAsset
    
    ```
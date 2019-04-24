1. Go to projections.py and specify the required parameters
2. Run projections.py 
    ```shell
    python projections.py
    ```
3. Import the function that simulates the tomography signals from anywhere

    ```python
    from tomographySignalSimulation.simulateSignal import simulate_signal 
    ``` 
4. Call the function
    ```python
    simulate_signal(emissivity, "projections")
    ```
See projections.py and simulateSignal.py docstrings for more information
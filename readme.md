1. Go to cameras.py and specify the required parameters
2. Run cameras.py 
    ```shell
    python cameras.py
    ```
3. Go to projections.py and specify the required parameters
4. Run projections.py 
    ```shell
    python projections.py
    ```
5. Import the function that simulates the tomography signals from anywhere

    ```python
    from tomographySignalSimulation.simulateSignal import simulate_signal 
    ``` 
6. Call the function
    ```python
    simulate_signal(emissivity, "projections")
    ```
See the docstrings and comments on each file for more information
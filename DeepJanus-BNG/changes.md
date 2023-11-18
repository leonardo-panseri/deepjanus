## General
- Add _.gitignore_
- _requirements.txt_:
  - Remove shapely version
  - Remove outdated bundled beamngpy library, adding it here instead
  - Add missing required libraries
- Convert ML models from h5 format to standard Keras format, to ensure compatibility with new Python versions and fix a memory leak

## Udacity Integration
- _beamng_car_cameras.py_:
    - Modify cameras position to capture road ahead of vehicle (BeamNG coordinate system has changed and y-axis has been inverted)
    - Add references to BeamNGpy and Vehicle instances in constructor, as they are now required to instantiate Camera objects
    - Update parameters of Camera objects to new version of beamngpy
    - Set center Camera used for ML model input to use shared memory instead of polling the simulator

## Self Driving
- _beamng_brewer.py_:
  - Refactor brewer camera to use beamngpy Camera object and remove custom BeamNGCamera class
  - Move here initialization of car cameras and other sensors
  - Update call to open and connect to simulator to run only if simulator is open with new API
  - Convert angle of vehicle from euler to quaternion (needed by new beamngpy version)
  - Add framerate limit to _set_deterministic()_ call as it is now mandatory
- _beamng_nvidia_runner.py_:
  - Modify calls to brewer and vehicle state reader to accommodate for changes
  - Make simulation run for 1 step at start, otherwise Camera won't have data
  - Modify method to obtain image from Camera, since Camera is now using shared memory
- _beamng_tig_maps.py_:
  - Modify user content path for new version of BeamNG.tech
- _oob_monitor.py_
  - Fix incorrect return type annotation for _get_oob_info()_ function
- _simulation_data_collector.py_:
  - Change from BeamNGCamera to Camera in constructor
  - Modify _take_car_picture_if_needed()_ for new Camera API
- _vehicle_state_reader.py_:
  - Remove car sensors initialization

## Probabilistic
- Implement method to estimate probabilistic relevance of individuals at the frontier in _explore_neighborhood.py_:
  - Given the results of a DeepJanus experiment, load individuals at the frontier from their serialized representation and for each one of them
    1. Evaluate the two original members again and compare the evaluation with the one obtained in the original experiment (to check if experiments are deterministic and reproducible)
    2. Generate new members that are "neighbors" of the individual: for both members, mutate the same road by a different value
    3. Evaluate each member of the two neighborhoods
    4. Save the results of the simulations, along with the percentage of members of the neighborhoods that are outside the frontier
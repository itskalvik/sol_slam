To change parameters at runtime over the CLI:

```
ros2 param set /sonoptix_odometry solver.iterations 20
ros2 param set /sonoptix_odometry mapping.update_alpha 0.3
```

To trigger the reset over the CLI:

```
ros2 service call /sonoptix_odometry/reset_odometry std_srvs/srv/Trigger
```
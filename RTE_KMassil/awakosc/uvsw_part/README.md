Black-box package to run wake oscillator simulations

To run the example script (test/example.py), you should install the uvsw_part package first; the test/create_env.sh script let you create a virtualenv to install the uvsw_part package, its dependencies and other needed python packages:

```
$> cd test
$> ./create_env.sh
$> source .venv/bin/activate
$> python3 example.py
```

If you do not wish to use a virtualenv, you can install the packages using these commands:

```
$> cd uvsw_part
$> python3 setup.py install [--user]
$> cd test
$> pip3 install -r requirements.txt [--user]
$> python3 example.py
```

Should you make any edit to the uvsw_part package itself, you should reinstall it to benefit from these modifications (either in the virtualenv or not).

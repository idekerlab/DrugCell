
# ENV Lambda
# Environment settings for Lambda (Swift, Python, R, Tcl, etc.)

# Everything is installed in here:
SFW=/homes/woz/Public/sfw

SWIFT=$SFW/swift-t/2022-11-02
PY=$SFW/Anaconda
# EQPY=$SFW/EQ-Py
EQR=$SFW/EQ-R
R=$SFW/R-4.1.0

PATH=$SWIFT/stc/bin:$PATH
PATH=$PY/bin:$PATH

export LD_LIBRARY_PATH=$R/lib/R/lib:${LD_LIBRARY_PATH:-}

# How to run CANDLE models:
CANDLE_MODEL_IMPL="app"

# PYTHONPATH=$EQPY/src:${PYTHONPATH:-}

# Log settings to output
echo "Programs:"
which python swift-t | nl
# Cf. utils.sh
show     PYTHONHOME
log_path LD_LIBRARY_PATH

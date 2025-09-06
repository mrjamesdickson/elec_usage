source env/bin/activate
pip install -r requirements.txt
python3 elec_usage.py ./usage.csv --agg-include-projection

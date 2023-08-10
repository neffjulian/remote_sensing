
echo "SRCNN"
python src/main.py -c configs/srcnn.yaml

echo "RRDB"
python src/main.py -c configs/rrdb.yaml

echo "EDSR"
python src/main.py -c configs/edsr.yaml
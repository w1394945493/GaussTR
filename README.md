
SupperOcc


cd /home/lianghao/wangyushen/Projects/GaussTR/superocc/models/ops/tile_localagg_prob_sq
rm -rf build/
find . -name "*.so" -delete
pip install -e .

cd /home/lianghao/wangyushen/Projects/GaussTR/superocc/models/ops/msmv_sampling
rm -rf build/
find . -name "*.so" -delete
pip install -e .

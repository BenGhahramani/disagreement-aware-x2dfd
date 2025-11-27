TARGET=/data/250010183/workspace/X2DFD/src/forensics
SRC=/data/250010183/hetao/MLLMs/rebuttal_two_model
mkdir -p $TARGET/{networks,utils,weights/ours-sync}
cp -f $SRC/src/forensics/networks/__init__.py               $TARGET/networks/__init__.py
cp -f $SRC/src/forensics/networks/resnet_mod.py             $TARGET/networks/resnet_mod.py
cp -f $SRC/src/forensics/utils/processing.py                $TARGET/utils/processing.py
cp -f $SRC/weights/forensics_models/ours-sync/config.yaml   $TARGET/weights/ours-sync/config.yaml
cp -f $SRC/weights/forensics_models/ours-sync/ours-sync.pth $TARGET/weights/ours-sync/ours-sync.pth
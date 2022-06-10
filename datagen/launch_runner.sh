#!/bin/bash

datagen_id=$1 
echo "Launching blender datagen of id $datagen_id"

cd /media/beans/ssd/blenders_for_dataloader/blender_3.1_$datagen_id

new_blendfile_path=/media/beans/ssd/blenders_for_dataloader/tmp/fustbol_$datagen_id.blend

# prob not necessary, but making sure we don't overwrite the original blendfile
cp /media/beans/ssd/bespoke/datagen/blend_files/fustbol.blend $new_blendfile_path

./blender $new_blendfile_path --background --python /media/beans/ssd/bespoke/datagen/bpy_runner.py -- $datagen_id

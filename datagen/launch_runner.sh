#!/bin/bash

datagen_id=$1 
echo "Launching blender datagen of id $datagen_id"

cd /home/beans/blenders_for_dataloader/blender_3.6_$datagen_id

new_blendfile_path=/home/beans/blenders_for_dataloader/tmp/fustbol3_$datagen_id.blend

# prob not necessary, but making sure we don't overwrite the original blendfile
cp /home/beans/bespoke/datagen/blend_files/fustbol3.blend $new_blendfile_path

./blender $new_blendfile_path --background --python /home/beans/bespoke/datagen/bpy_runner.py -- $datagen_id

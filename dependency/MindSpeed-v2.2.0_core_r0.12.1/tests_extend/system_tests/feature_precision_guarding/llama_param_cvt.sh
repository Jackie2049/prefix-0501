#! /bin/bash
echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)
   VALUE="${ARGUMENT#*${KEY}=}"

   export "$KEY"="$VALUE"
   echo "$KEY=$VALUE"
done
echo "---------------------------------"


python ${PY_SCRIPT_PATH} \
      --input-model-dir ${INPUT_MODEL_DIR} \
      --output-model-dir ${OUTPUT_MODEL_DIR} \
      --tensor-model-parallel-size ${TP_SIZE} \
      --pipeline-model-parallel-size ${PP_SIZE} \
       ${VP_SIZE:+--num-layers-per-virtual-pipeline-stage "$VP_SIZE"} \
       ${SWIGLU:+--swiglu} \
      --num-layers ${NUM_LAYERS}

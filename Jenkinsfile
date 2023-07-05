import org.apache.commons.io.FilenameUtils
import groovy.json.JsonOutput

def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME" || true
        lsb_release -sd || true
        uname -r || true
        cat /sys/module/amdgpu/version || true
        ls /opt/ -la || true
    """
}

//makes sure multiple builds are not triggered for branch indexing
def resetbuild() {
    prev_build = currentBuild.getPreviousBuild()
    
    if (prev_build != null && prev_build.getBuildCauses().toString().contains('BranchIndexingCause')) {
        def buildNumber = BUILD_NUMBER as int; 
        if (buildNumber > 1) milestone(buildNumber - 1); 
        milestone(buildNumber)
    }
}

def run_uts() {
    sh 'pytest --verbose' 
}
def run_pytorch_framework() {
    sh 'pip istall -r examples/pytorch/_test_requirements.txt'
    sh 'pytest examples/pytorch/test_pytorch_examples.py --verbose'
    sh 'pytest examples/pytorch/test_accelerate_examples.py --verbose'
    sh 'pytest examples/pytorch/test_xla_examples.py --verbose'
}
def run_flax_framework(){
    sh 'pip istall -r examples/flax/_test_requirements.txt'
    sh 'pytest examples/flax/test_flax_examples.py --verbose'
}
def run_tensorflow_framework(){
    sh 'pip istall -r examples/tensorflow/_test_requirements.txt'
    sh 'pytest examples/tensorflow/test_tensorflow_examples.py --verbose' 
}


def run_pytorch_pipeline(model, pipeline){

    switch case (pipeline) {
        case 'audio-classification':
            //facebook/wav2vec2-base
            sh 'pip install -r examples/pytorch/audio-classification/requirements.txt'
            sh '''
            python examples/pytorch/audio-classification/run_audio_classification.py \
                --model_name_or_path ${model} \
                --dataset_name superb \
                --dataset_config_name ks \
                --output_dir wav2vec2-base-ft-keyword-spotting \
                --overwrite_output_dir \
                --remove_unused_columns False \
                --do_train \
                --do_eval \
                --fp16 \
                --learning_rate 3e-5 \
                --max_length_seconds 1 \
                --attention_mask False \
                --warmup_ratio 0.1 \
                --num_train_epochs 5 \
                --per_device_train_batch_size 32 \
                --gradient_accumulation_steps 4 \
                --per_device_eval_batch_size 32 \
                --dataloader_num_workers 4 \
                --logging_strategy steps \
                --logging_steps 10 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end True \
                --metric_for_best_model accuracy \
                --save_total_limit 3 \
                --seed 0 
            '''
            break
        case 'image-classification':
            sh 'pip install -r examples/pytorch/image-classification/requirements.txt'
            sh '''
            python examples/pytorch/image-classification/run_image_classification.py \
                --model_name_or_path ${model} \
                --dataset_name beans \
                --output_dir ./beans_outputs/ \
                --remove_unused_columns False \
                --do_train \
                --do_eval \
                --learning_rate 2e-5 \
                --num_train_epochs 5 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --logging_strategy steps \
                --logging_steps 10 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end True \
                --save_total_limit 3 \
                --seed 1337
            '''
            break
        case 'language-modeling':
            sh 'pip install -r examples/pytorch/language-modeling/requirements.txt'
            sh '''
            python examples/pytorch/language-modeling/run_clm.py \
                --model_name_or_path ${model} \
                --dataset_name wikitext \
                --dataset_config_name wikitext-2-raw-v1 \
                --output_dir ./output \
                --do_train \
                --do_eval \
                --learning_rate 2e-5 \
                --num_train_epochs 5 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --logging_strategy steps \
                --logging_steps 10 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end True \
                --save_total_limit 3 \
                --seed 1337
            '''
            break
        case 'question-answering':
            sh 'pip install -r examples/pytorch/question-answering/requirements.txt'
            sh '''
            python examples/pytorch/question-answering/run_qa.py \
                --model_name_or_path ${model} \
                --dataset_name squad \
                --output_dir ./output \
                --do_train \
                --do_eval \
                --learning_rate 2e-5 \
                --num_train_epochs 5 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --logging_strategy steps \
                --logging_steps 10 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end True \
                --save_total_limit 3 \
                --seed 1337
            '''
            break
        case 'summarization':
            sh 'pip install -r examples/pytorch/summarization/requirements.txt'
            sh '''
            python examples/pytorch/summarization/run_summarization.py \
                --model_name_or_path ${model} \
                --dataset_name cnn_dailymail \
                --output_dir ./output \
                --do_train \
                --do_eval \
                --learning_rate 2e-5 \
                --num_train_epochs 5 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --logging_strategy steps \
                --logging_steps 10 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end True \
                --save_total_limit 3 \
                --seed 1337
            '''
            break
        case 'text-classification':
            sh 'pip install -r examples/pytorch/text-classification/requirements.txt'
            sh '''
            python examples/pytorch/text-classification/run_glue.py \
                --model_name_or_path ${model} \
                --dataset_name imdb \
                --output_dir ./output \
                --do_train \
                --do_eval \
                --learning_rate 2e-5 \
                --num_train_epochs 5 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --logging_strategy steps \
                --logging_steps 10 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end True \
                --save_total_limit 3 \
                --seed 1337
            '''
            break   
        case 'translation':
            sh 'pip install -r examples/pytorch/translation/requirements.txt'
            sh '''
            python examples/pytorch/translation/run_translation.py \
                --model_name_or_path ${model} \
                --do_train \
                --do_eval \
                --source_lang en \
                --target_lang ro \
                --dataset_name wmt16 \
                --dataset_config_name ro-en \
                --output_dir /tmp/tst-translation \
                --per_device_train_batch_size=4 \
                --per_device_eval_batch_size=4 \
                --overwrite_output_dir \
                --predict_with_generate
            '''
            break
        case 'speech-pretraining':
            sh 'pip install -r examples/pytorch/speech-pretraining/requirements.txt'
            sh '''
            python examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py \
                --model_name_or_path ${model} \
                --dataset_name="librispeech_asr" \
                --dataset_config_names clean clean \
                --dataset_split_names validation test \
                --model_name_or_path="patrickvonplaten/wav2vec2-base-v2" \
                --output_dir="./wav2vec2-pretrained-demo" \
                --max_train_steps="20000" \
                --num_warmup_steps="32000" \
                --gradient_accumulation_steps="8" \
                --learning_rate="0.005" \
                --weight_decay="0.01" \
                --max_duration_in_seconds="20.0" \
                --min_duration_in_seconds="2.0" \
                --logging_steps="1" \
                --saving_steps="10000" \
                --per_device_train_batch_size="8" \
                --per_device_eval_batch_size="8" \
                --adam_beta1="0.9" \
                --adam_beta2="0.98" \
                --adam_epsilon="1e-06" \
                --gradient_checkpointing \
                --mask_time_prob="0.65" \
                --mask_time_length="10"
            '''
            break
        case 'image-pretraining':
            sh 'pip install -r examples/pytorch/image-pretraining/requirements.txt'
            sh '''
            python examples/pytorch/image-pretraining/run_mim.py \
                --model_type ${model} \
                --output_dir ./outputs/ \
                --overwrite_output_dir \
                --remove_unused_columns False \
                --label_names bool_masked_pos \
                --do_train \
                --do_eval \
                --learning_rate 2e-5 \
                --weight_decay 0.05 \
                --num_train_epochs 100 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --logging_strategy steps \
                --logging_steps 10 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end True \
                --save_total_limit 3 \
                --seed 1337
            '''
            break
        case 'speech-recognition':
            sh 'pip install -r examples/pytorch/speech-recognition/requirements.txt'
            sh '''
            python examples/pytorch/speech-recognition/run_speech_recognition_ctc.py \
                --dataset_name="common_voice" \
                --model_name_or_path=${model} \
                --dataset_config_name="tr" \
                --output_dir="./wav2vec2-common_voice-tr-demo" \
                --overwrite_output_dir \
                --num_train_epochs="15" \
                --per_device_train_batch_size="16" \
                --gradient_accumulation_steps="2" \
                --learning_rate="3e-4" \
                --warmup_steps="500" \
                --evaluation_strategy="steps" \
                --text_column_name="sentence" \
                --length_column_name="input_length" \
                --save_steps="400" \
                --eval_steps="100" \
                --layerdrop="0.0" \
                --save_total_limit="3" \
                --freeze_feature_encoder \
                --gradient_checkpointing \
                --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” � \
                --fp16 \
                --group_by_length \
                --do_train --do_eval 
            '''
            break
        case 'text-generation':
            sh 'pip install -r examples/pytorch/text-generation/requirements.txt'
            sh '''
            python examples/pytorch/text-generation/run_generation.py \
                --model_type=gpt2 \
                --model_name_or_path=${model}
            '''
            break
        case 'contrastive-image-text':
            sh 'pip install -r examples/pytorch/contrastive-image-text/requirements.txt'
            sh 'mkdir data'
            sh 'cd data'
            sh 'wget http://images.cocodataset.org/zips/train2017.zip'
            sh 'wget http://images.cocodataset.org/zips/val2017.zip'
            sh 'wget http://images.cocodataset.org/zips/test2017.zip'
            sh 'wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
            sh 'wget http://images.cocodataset.org/annotations/image_info_test2017.zip'
            sh 'cd ..'
            sh '''
            python examples/pytorch/contrastive-image-text/run_clip.py \
                --output_dir ./clip-roberta-finetuned \
                --model_name_or_path ${model} \
                --data_dir $PWD/data \
                --dataset_name ydshieh/coco_dataset_script \
                --dataset_config_name=2017 \
                --image_column image_path \
                --caption_column caption \
                --remove_unused_columns=False \
                --do_train  --do_eval \
                --per_device_train_batch_size="64" \
                --per_device_eval_batch_size="64" \
                --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
                --overwrite_output_dir 
            '''
            break
        case 'semantic-segmentation':
            sh 'pip install -r examples/pytorch/semantic-segmentation/requirements.txt'
            sh '''
            python examples/pytorch/semantic-segmentation/run_semantic_segmentation.py \
                --model_name_or_path ${model} \
                --dataset_name segments/sidewalk-semantic \
                --output_dir ./segformer_outputs/ \
                --remove_unused_columns False \
                --do_train \
                --do_eval \
                --evaluation_strategy steps \
                --max_steps 10000 \
                --learning_rate 0.00006 \
                --lr_scheduler_type polynomial \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --logging_strategy steps \
                --logging_steps 100 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --seed 1337
            '''
            break
        case 'token-classification':
            sh 'pip install -r examples/pytorch/token-classification/requirements.txt'
            sh '''
            python examples/pytorch/token-classification/run_ner.py \
                --model_name_or_path ${model} \
                --dataset_name conll2003 \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        case 'multiple-choice':
            sh 'pip install -r examples/pytorch/multiple-choice/requirements.txt'
            sh '''
            python examples/pytorch/multiple-choice/run_swag.py \
                --model_name_or_path ${model} \
                --do_train \
                --do_eval \
                --learning_rate 5e-5 \
                --num_train_epochs 3 \
                --output_dir /tmp/swag_base \
                --per_gpu_eval_batch_size=16 \
                --per_device_train_batch_size=16 \
                --overwrite_output
            '''
            break
        case 'benchmarking':
            sh 'pip install -r examples/pytorch/benchmarking/requirements.txt'
            sh '''
            python examples/pytorch/benchmarking/run_benchmark.py \
                --models ${model} \
                --inference \
                --cuda 
                --fp16 
                --training 
                --verbose 
                --speed 
                --memory
                --trace_memory_line_by_line
                --log_print 
                --env_print
            '''
            break
        default:
            echo "Pytorch does not support this pipeline type ${pipeline}"
    }

}

def run_flax_pipeline(model, pipeline){

    switch case (pipeline) {
            case 'language-modeling':
                sh 'pip install -r examples/flax/language-modeling/requirements.txt'
                sh '''
                python examples/flax/language-modeling/run_clm_flax.py \
                    --model_name_or_path ${model} \
                    --dataset_name wikitext \
                    --dataset_config_name wikitext-2-raw-v1 \
                    --output_dir ./output \
                    --do_train \
                    --do_eval 
                '''
                break
            case 'question-answering':
                sh 'pip install -r examples/flax/question-answering/requirements.txt'
                sh '''
                python examples/flax/question-answering/run_qa.py \
                    --model_name_or_path ${model} \
                    --dataset_name squad \
                    --dataset_config_name squad \
                    --output_dir ./output \
                    --do_train \
                    --do_eval 
                '''
                break
            case 'summarization':
                sh 'pip install -r examples/flax/summarization/requirements.txt'
                sh '''
                python examples/flax/summarization/run_summarization_flax.py \
                    --model_name_or_path ${model} \
                    --tokenizer_name ${model} \
                    --dataset_name="xsum" \
                    --do_train --do_eval --do_predict --predict_with_generate \
                    --num_train_epochs 6 \
                    --learning_rate 5e-5 --warmup_steps 0 \
                    --per_device_train_batch_size 64 \
                    --per_device_eval_batch_size 64 \
                    --overwrite_output_dir \
                    --max_source_length 512 --max_target_length 64 
                '''
                break
            case 'text-classification':
                sh 'pip install -r examples/flax/text-classification/requirements.txt'
                sh '''
                python examples/flax/text-classification/run_glue_flax.py \
                    --model_name_or_path ${model} \
                    --task_name mrpc \
                    --max_seq_length 128 \
                    --learning_rate 2e-5 \
                    --num_train_epochs 3 \
                    --per_device_train_batch_size 4 \
                    --eval_steps 100 \
                    --output_dir ./output \
                    --do_train \
                    --do_eval 
                '''
                break
            case 'token-classification':
                sh 'pip install -r examples/flax/token-classification/requirements.txt'
                sh '''
                python examples/flax/token-classification/run_flax_ner.py \
                    --model_name_or_path ${model} \
                    --dataset_name conll2003 \
                    --output_dir ./output \
                    --do_train \
                    --do_eval 
                '''
                break
            case 'image-captioning':
                sh 'pip install -r examples/flax/image-captioning/requirements.txt'
                sh '''
                python examples/flax/image-captioning/run_image_captioning_flax.py \
                    --model_name_or_path ${model} \
                    --output_dir ./output \    
                	--dataset_name ydshieh/coco_dataset_script \
                    --dataset_config_name=2017 \
                    --image_column image_path \
                    --caption_column caption \
                    --do_train --do_eval --predict_with_generate \
                    --num_train_epochs 1 \
                    --eval_steps 500 \
                    --learning_rate 3e-5 --warmup_steps 0 \
                    --overwrite_output_dir \
                    --max_target_length 32 \
                    --logging_steps 10 \
                    --block_size 16384 
                '''
                break
            case 'vision':
                sh 'pip install -r examples/flax/vision/requirements.txt'
                sh '''
                python examples/flax/vision/run_image_classification.py \
                    --model_name_or_path ${model} \
                    --dataset_name imagenet2012 \
                    --output_dir ./output \
                    --do_train \
                    --do_eval 
                '''
                break   
            default:
                echo "Flax does not support this pipeline type ${pipeline}"
    }

}


def run_tensorflow_pipeline(model, pipeline){

    switch case (pipeline) {
        case 'text-classification':
            sh 'pip install -r examples/tensorflow/text-classification/requirements.txt'
            sh '''
            python examples/tensorflow/text-classification/run_text_classification.py \
                --model_name_or_path ${model} \
                --dataset_name imdb \
                --dataset_config_name imdb \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        case 'translation':
            sh 'pip install -r examples/tensorflow/translation/requirements.txt'
            sh '''
            python examples/tensorflow/translation/run_translation.py \
                --model_name_or_path ${model} \
                --dataset_name wmt16 \
                --dataset_config_name ro-en \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        case 'language-modeling':
            sh 'pip install -r examples/tensorflow/language-modeling/requirements.txt'
            sh '''
            python examples/tensorflow/language-modeling/run_clm.py \
                --model_name_or_path ${model} \
                --dataset_name wikitext \
                --dataset_config_name wikitext-2-raw-v1 \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        case 'multiple-choice':
            sh 'pip install -r examples/tensorflow/multiple-choice/requirements.txt'
            sh '''
            python examples/tensorflow/multiple-choice/run_swag.py \
                --model_name_or_path ${model} \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        case 'question-answering':
            sh 'pip install -r examples/tensorflow/question-answering/requirements.txt'
            sh '''
            python examples/tensorflow/question-answering/run_qa.py \
                --model_name_or_path ${model} \
                --dataset_name squad \
                --dataset_config_name squad \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        case 'summarization':
            sh 'pip install -r examples/tensorflow/summarization/requirements.txt'
            sh '''
            python examples/tensorflow/summarization/run_summarization.py \
                --model_name_or_path ${model} \
                --dataset_name cnn_dailymail \
                --dataset_config_name "3.0.0" \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        case 'token-classification':
            sh 'pip install -r examples/tensorflow/token-classification/requirements.txt'
            sh '''
            python examples/tensorflow/token-classification/run_ner.py \
                --model_name_or_path ${model} \
                --dataset_name conll2003 \
                --dataset_config_name conll2003 \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        case 'benchmarking':
            sh 'pip install -r examples/tensorflow/benchmarking/requirements.txt'
            sh '''
            python examples/tensorflow/benchmarking/run_benchmark_tf.py \
                --model_name_or_path ${model} \
                --dataset_name squad \
                --dataset_config_name squad \
                --output_dir ./output \
                --do_train \
                --do_eval 
            '''
            break
        default:    
            echo "Tensorflow does not support this pipeline type ${pipeline}"
    }

}


def runTest(model, framework, pipeline) {
    try {
        // get the node info
        show_node_info()

        run_uts() 
        // selection of hf hub run params.MODEL, params.FRAMEWORK, params.PIPELINE
        if (framework.contains('all')) { // if model pipeline selection 
            run_pytorch_framework() 
            run_flax_framework() 
            run_tensorflow_framework()
            run_pytorch_pipeline(model, pipeline)
            run_flax_pipeline(model, pipeline)
            run_tensorflow_pipeline(model, pipeline)
        }
        else if (framework.contains('pytorch')) {
            run_pytorch_framework() 
            run_pytorch_pipeline(model, pipeline)
        }
        else if (framework.contains('flax')) {
            run_flax_framework() 
            run_flax_pipeline(model, pipeline)
        }
        else if (framework.contains('tensorflow')) {
            run_tensorflow_framework() 
            run_tensorflow_pipeline(model, pipeline)
        }
        else {
            echo "No framework selected"
            run_pytorch_framework() 
            run_flax_framework() 
            run_tensorflow_framework()
        }

        echo "SUCCESS: ${model_info.name}-${arch} completed successfully."

    } catch (e) {
        currentBuild.result = 'FAILURE'
        error "${model}-${pipeline}-${framework} threw \"${e}\"."
    }

}

//TODO fill in jobs run in parallel each test // add test selection here with params 
def tryTests(model, framework, pipeline) {
    jobs["${model}-${pipeline}-${framework}"] = {
        // protect inside catchError to set stageResult correctly
        catchError(message: 'Caught runTest error; continuing', stageResult: 'FAILURE') {
            stage("${model}-${pipeline}-${framework}") {
                gitStatusWrapper(credentialsId: params.GITHUB_CREDENTIALS, gitHubContext: "Jenkins - ${model}-${pipeline}-${framework}", account: 'ROCmSoftwarePlatform', repo: 'Transformers') {
                    // GPUs cannot be shared amongst stages on the same machine
                    lock("${env.NODE_NAME}".trim()) {
                        runTest(model, framework, pipeline)
                    }
                }
            }
        } 

    }
}

def doSteps() {
    def targetNode = params.NODE_OVERRIDE != '' ? "${params.NODE_OVERRIDE}" : "dlmodels && ${arch}" 
    // Record the value of targetNode
    echo "The value of targetNode is: ${targetNode}"
    node(targetNode) {
        //show node information
        show_node_info()
        
        // Clean before build
        cleanWs()

        // We need to explicitly checkout from SCM here
        checkout scm

        withPythonEnv('python3') {
            // install requirements
            sh 'python3 -m pip install --upgrade pip || true'
            sh 'python3 -m pip install -r requirements.txt || true'
        }

        // selftest
        catchError(message: 'Caught selftest error; continuing', stageResult: 'FAILURE') {
            stage("selftest") {
                if ( params.TFS_UT ) {
                    gitStatusWrapper(credentialsId: params.GITHUB_CREDENTIALS, gitHubContext: "Jenkins - selftest", account: 'ROCmSoftwarePlatform', repo: 'Transformers') {
                        withPythonEnv('python3') {
                            sh 'python3 -m pytest'
                        }
                    }
                }
            }
        }

        // TODO test repo and make function for each framework test and run selected 
        jobs = [:]        
        withPythonEnv('python3') {
            tryTests(params.MODEL, params.FRAMEWORK, params.PIPELINE)
        }

        // run stages in parrallel
        withPythonEnv('python3') {
            parallel(jobs)
        }

    }
}

//steps: always run pytest and framework tests 
// TODO add hardware spesific tests
// add options to run specific pipeline

pipeline {
    agent { label 'master' }

    // set main branch
    environment {
        MAIN_BRANCH = 'main'
        GITHUB_CREDENTIALS = credentials('github-creds')
        
    }

    parameters {
        string(name: 'PIPELINE', defaultValue: '', description: 'Specify Transformer pipeline type. (default valeu to be selected TODO) Possible values (qa, )')
        string(name: 'TARGET_ARCH', defaultValue: '', description: 'Run transformers on specific arch. Blank will run on any dlmodels node if non-main branch and on (gfx906, MI250, gfx908) if main branch. Use comma-separated arch for targeting multiple arch. Supported arch types - gfx906, MI250, MI250X-A1, gfx908, dlmodels')
        string(name: 'MODEL', defaultValue: '', description: 'Run a model from huggingface hub with pipeline must be compatible (default to be selected TODO).')
        string(name: 'FRAMEWROK', defaultValue: 'pytoch', description: 'FRAMEWORK to run. The default value (empty) runs pytorch.')
        string(name: 'NODE_OVERRIDE', defaultValue: '', description: 'Node override to target a Jenkins node directly. Warning : This will override TARGET_ARCH setting.')
        booleanParam(name: 'TFS_UT', defaultValue: true, description: 'Run Transformer self-test prior to running model') 
    }

    triggers {
         parameterizedCron( env.BRANCH_NAME == 'main' ? '''
            @midnight %FRAMEWROK=pytorch;TFS_UT=true
            0 02 * * 1 %FRAMEWROK=flax;TFS_UT=true
            0 02 * * 2 %FRAMEWROK=tensorflow;TFS_UT=true
            0 02 * * 3 %FRAMEWROK=flax;TFS_UT=true
            0 02 * * 4 %FRAMEWROK=tensorflow;TFS_UT=true
            0 02 * * 5 %FRAMEWROK=flax;TFS_UT=true
            0 02 * * 6 %FRAMEWROK=tensorflow;TFS_UT=truee
            0 02 * * 7 %FRAMEWROK=flax;TFS_UT=true
        ''' : "" )
    }

    stages {
        stage('resetbuild') {
            when { not { branch 'main' } } //makes sure it doesn't run on main branch
            steps {
                resetbuild()
            }
        }
        stage('the-matrix') {
            matrix {
                when { 
                    expression { 
                        (env.TARGET_ARCH == '' ) && (
                            (env.BRANCH_NAME == env.MAIN_BRANCH) && (
                                env.arch == 'gfx906' || 
                                env.arch == 'MI250' || 
                                env.arch == 'gfx908'  
                                ) ||
                            (env.BRANCH_NAME != env.MAIN_BRANCH) && (
                                env.arch == 'gfx906' )
                        ) || 
                        ( env.arch in params.TARGET_ARCH.replaceAll("\\s","").tokenize(","))  
                    }
                }
                axes {
                    axis {
                        name 'arch'
                        values 'gfx906', 'MI250', 'MI250_CA', 'MI250X-A1', 'A100', 'gfx908', 'dlmodels'
                    }
                }
                stages {
                    stage('tests/pipeline') {
                        steps {
                            script {
                                doSteps()
                            }
                        }
                    }
                }
            }
        }
    }
}


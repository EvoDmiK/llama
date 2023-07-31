#!/bin/bash
ROOT_PATH=/home/jovyan/dove/projects/llama2/llama

check=`ps -ef | grep app | wc | awk '{print $1}'`
if [ $check -gt 1 ]; then
    echo "현재 gradio가 실행 중입니다."
    
else

    LOG_PATH=$ROOT_PATH/logs
    
    echo "실행코드가 있는 곳으로 이동 \n"
    cd $ROOT_PATH/gradioApp
    
    echo 현재 작업 디렉토리 입니다.
    pwd
    
    TODAY=$(date "+%Y-%m-%d")
    YESTERDAY=$(date -d "yesterday" "+%Y-%m-%d")
    echo "\n오늘 날짜 입니다." $TODAY
    
    mkdir -p $LOG_PATH/${YESTERDAY}_logs
    mv $LOG_PATH/${YESTERDAY}*.* $LOG_PATH/${YESTERDAY}_logs
    zip -r $LOG_PATH/${YESTERDAY}_logs.zip $LOG_PATH/${YESTERDAY}_logs
    rm -rf $LOG_PATH/${YESTERDAY}_logs
    
    if [ -d $LOG_PATH ]; then
        
        echo "\n로그를 저장할 폴더가 존재합니다.\n"
        idx=$(ls -l $LOG_PATH | grep $TODAY | grep ^- | wc -l)
        
        while [ ${#idx} -ne 3 ]; do
            idx="0"$idx
        done
        
    else
        mkdir -p $LOG_PATH
        echo "로그를 저장할 폴더가 생성되었습니다."
        idx="000"
    fi
    LOG_PATH=$LOG_PATH/${TODAY}_gradio_${idx}.log
    echo 이번에 생성할 로그 파일 경로 입니다. ${LOG_PATH}
    
    nohup python $ROOT_PATH/gradioApp/app.py > $LOG_PATH &
fi
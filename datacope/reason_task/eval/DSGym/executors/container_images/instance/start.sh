rm -rf logs
mkdir logs

nohup env PORT=9632 python container_main.py > logs/app1.log 2>&1 &
nohup env PORT=9633 python container_main.py > logs/app2.log 2>&1 &
nohup env PORT=9634 python container_main.py > logs/app3.log 2>&1 &
nohup env PORT=9635 python container_main.py > logs/app4.log 2>&1 &
nohup env PORT=9636 python container_main.py > logs/app5.log 2>&1 &
nohup env PORT=9637 python container_main.py > logs/app6.log 2>&1 &
nohup env PORT=9638 python container_main.py > logs/app7.log 2>&1 &
nohup env PORT=9639 python container_main.py > logs/app8.log 2>&1 &
nohup env PORT=9640 python container_main.py > logs/app9.log 2>&1 &
nohup env PORT=9641 python container_main.py > logs/app10.log 2>&1 &
nohup env PORT=9642 python container_main.py > logs/app11.log 2>&1 &
nohup env PORT=9643 python container_main.py > logs/app12.log 2>&1 &
nohup env PORT=9644 python container_main.py > logs/app13.log 2>&1 &
nohup env PORT=9645 python container_main.py > logs/app14.log 2>&1 &
nohup env PORT=9646 python container_main.py > logs/app15.log 2>&1 &
nohup env PORT=9647 python container_main.py > logs/app16.log 2>&1 &
nohup env PORT=9648 python container_main.py > logs/app17.log 2>&1 &
nohup env PORT=9649 python container_main.py > logs/app18.log 2>&1 &
nohup env PORT=9650 python container_main.py > logs/app19.log 2>&1 &
nohup env PORT=9651 python container_main.py > logs/app20.log 2>&1 &
nohup env PORT=9652 python container_main.py > logs/app21.log 2>&1 &
nohup env PORT=9653 python container_main.py > logs/app22.log 2>&1 &
nohup env PORT=9654 python container_main.py > logs/app23.log 2>&1 &
nohup env PORT=9655 python container_main.py > logs/app24.log 2>&1 &
nohup env PORT=9656 python container_main.py > logs/app25.log 2>&1 &
nohup env PORT=9657 python container_main.py > logs/app26.log 2>&1 &
nohup env PORT=9658 python container_main.py > logs/app27.log 2>&1 &
nohup env PORT=9659 python container_main.py > logs/app28.log 2>&1 &
nohup env PORT=9660 python container_main.py > logs/app29.log 2>&1 &
nohup env PORT=9661 python container_main.py > logs/app30.log 2>&1 &
nohup env PORT=9662 python container_main.py > logs/app31.log 2>&1 &
nohup env PORT=9663 python container_main.py > logs/app32.log 2>&1 &

ps aux | grep container_main.py

# use the following command to stop all the processes
# pkill -f "container_main.py"
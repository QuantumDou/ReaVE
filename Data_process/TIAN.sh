#!/bin/bash -l
#SBATCH --output=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Data_process/test-%j.log
#SBATCH --error=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Data_process/test-%j.log
#SBATCH --job-name=iu
#SBATCH --partition=interruptible_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128            # 对齐你的 srun 参数，申请 128 核
#SBATCH --mem=600G                     # 对齐你的 srun 参数
#SBATCH --time=1-00:00                 # 保持 2 天的时间限制（比 12 小时更稳）
#SBATCH --mail-user=1312438276@qq.com   # email address
#SBATCH --mail-type=END   #get email when job ends
#SBATCH --mail-type=FAIL    #get email when job aborts


RUNPATH=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Data_process
# RUNPATH=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/mimic_code

# RUNPATH=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Model
cd $RUNPATH

# activate virtualenv 
#source $HOME/TIAN_env/bin/activate
source activate inter2.5


export NUM_WORKERS=120
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1



python process_chest_imagenome.py




#python TD_HEUS_test.py ./data/worldcup/ego_list_1.pkl 15
#python TD_HEU_test.py ./data/worldcup/ego_list_1.pkl 15

# cpu, long_cup
#shared,nms_research,long_nms_research

#---- ILP

#5,10,15,20,25,30,35
#python MR_DP_EXACT_SC.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_DP_EXACT_SC_dblp_5_m.txt 2 40 

#python MR_T_EXACT_SC.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_T_EXACT_SC_dblp_5_m.txt 2 35

#python MR_TD_EXACT_WSC.py ./weighted_data/dblp/ego_list_5.pkl ./weighted_data/dblp/MR_TD_EXACT_WSC_dblp_5_m.txt 2 20 

#python MR_DP_EXACT_WSC.py ./weighted_data/dblp/ego_list_5.pkl ./weighted_data/dblp/MR_DP_EXACT_WSC_dblp_5_m.txt 2 20


#------WJM

#python MR_T_GWSC.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_T_GWSC_k.txt 2,10,30,50,70,90,110,130,150,170,190
#python MR_T_GWSC.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_T_GWSC_m.txt 2 10,30,50,70,90,110,130,150,170,190

#python MR_DP_GWSC.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_k.txt 190
#python MR_DP_GWSC.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_k.txt 2,10,30,50,70,90,110,130,150,170,190
#python MR_DP_GWSC.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_m.txt 2 10,30,50,70,90,110,130,150,170,190
#python MR_DP_GWSC.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_m.txt 2 10


#python MR_T_GWSC_q.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_T_GWSC_q_k.txt 2,10,30,50,70,90,110,130,150,170,190 10
#python MR_T_GWSC_q.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_T_GWSC_q_m.txt 2 10,30,50,70,90,110,130,150,170,190 10

#python MR_DP_GWSC_q.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_q_k.txt 190 10
#python MR_DP_GWSC_q.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_q_k.txt 2,10,30,50,70,90,110,130,150,170,190 10
#python MR_DP_GWSC_q.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_q_m.txt 2 10,30,50,70,90,110,130,150,170,190 10


#python MR_T_GWSC_d.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_T_GWSC_d_k.txt 2,10,30,50,70,90,110,130,150,170,190 10
#python MR_T_GWSC_d.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_T_GWSC_d_m.txt 2 10,30,50,70,90,110,130,150,170,190 10


#python MR_DP_GWSC_d.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_d_k.txt 2,10,30,50,70,90,110,130,150,170,190 10
#python MR_DP_GWSC_d.py ./weighted_data/mathoverflow/ego_list_10.pkl ./weighted_data/mathoverflow/MR_DP_GWSC_d_m.txt 2 10,30,50,70,90,110,130,150,170,190 10


#—----------- WOC


#python MR_TD_HEU.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/MR_TD_HEU_k.txt 700
#python MR_TD_HEU.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/MR_TD_HEU_k.txt 2,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700
#python MR_TD_HEU.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/MR_TD_HEU_m.txt 2 10,50,100,150,200,250,300,350,400,450,500,550,600,650,700
#python MR_TD_HEU.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/MR_TD_HEU_m.txt 2 700


#python MR_TD_HEUS.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/MR_TD_HEUS_k.txt 700
#python MR_TD_HEUS.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/MR_TD_HEUS_k.txt 2,10,50,100,150,200,250,300,350,400,450,500,550,600,650,700
#python MR_TD_HEUS.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/MR_TD_HEUS_m.txt 2 10,50,100,150,200,250,300,350,400,450,500,550,600,650,700
#python MR_TD_HEUS.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/MR_TD_HEUS_m.txt 2 700

#---------------
#python MR_TD_HEU.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_MR_TD_HEU_5_k.txt 2,5,10,15,20,25,30,35
#python MR_TD_HEU.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_MR_TD_HEU_5_m.txt 2 5,10,15,20,25,30,35,38

#python MR_TD_HEUS.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_TD_HEUS_5_k.txt 2,5,10,15,20,25,30,35
#python MR_TD_HEUS.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_TD_HEUS_5_m.txt 2 5,10,15,20,25,30,35,38

#python MR_DP_G_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_DP_G_SCUB_5_k.txt 2,5,10,15,20,25,30,35
#python MR_DP_G_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_DP_G_SCUB_5_m.txt 2 5,10,15,20,25,30,35,38

#python MR_TD_G_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_TD_G_SCUB_5_k.txt 2,5,10,15,20,25,30,35
#python MR_TD_G_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_TD_G_SCUB_5_m.txt 2 5,10,15,20,25,30,35,38

#python MR_DP_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_DP_EXACT_SCUB_5_k.txt 2,5,10,15,20,25,30,35
#python MR_DP_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_DP_EXACT_SCUB_5_m.txt 2 5,10,15,20,25,30,35,38

#python MR_TD_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_TD_EXACT_SCUB_5_k.txt 2,5,10,15,20,25,30,35
#python MR_TD_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_TD_EXACT_SCUB_5_m.txt 2 5,10,15,20,25,30,35,38

#python MR_DP_EXACT_SC.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_DP_EXACT_SC_5_k.txt 2,5,10,15,20,25,30,35
#python MR_DP_EXACT_SC.py ./data/dblp/ego_list_5.pkl ./data/dblp/MR_dblp_DP_EXACT_SC_5_m.txt 2 5,10,15,20,25,30,35,38
#---------------

#python DP_EXACT_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_DP_EXACT_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python DP_EXACT_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_DP_EXACT_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python DP_EXACT_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_DP_EXACT_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python DP_EXACT_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_DP_EXACT_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python DP_EXACT_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_DP_EXACT_SCUB_5.txt ./data/fb-friends/label_5.txt 8

#python DP_EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_DP_EXACT_SCUB_1.txt ./data/dblp/label_1.txt 5
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_DP_EXACT_SCUB_2.txt ./data/dblp/label_2.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_DP_EXACT_SCUB_3.txt ./data/dblp/label_3.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_DP_EXACT_SCUB_4.txt ./data/dblp/label_4.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_DP_EXACT_SCUB_5.txt ./data/dblp/label_5.txt 8

#python DP_EXACT_SCUB.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_DP_EXACT_SCUB_1.txt ./data/STACKEXCH/label_1.txt 22
#python DP_EXACT_SCUB.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_DP_EXACT_SCUB_2.txt ./data/STACKEXCH/label_2.txt 22
#python DP_EXACT_SCUB.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_DP_EXACT_SCUB_3.txt ./data/STACKEXCH/label_3.txt 22
#python DP_EXACT_SCUB.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_DP_EXACT_SCUB_4.txt ./data/STACKEXCH/label_4.txt 16
#python DP_EXACT_SCUB.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_DP_EXACT_SCUB_5.txt ./data/STACKEXCH/label_5.txt 22

#python DP_EXACT_SCUB.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_DP_EXACT_SCUB_1.txt ./data/stackoverflow/label_1.txt 22
#python DP_EXACT_SCUB.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_DP_EXACT_SCUB_2.txt ./data/stackoverflow/label_2.txt 22
#python DP_EXACT_SCUB.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_DP_EXACT_SCUB_3.txt ./data/stackoverflow/label_3.txt 22
#python DP_EXACT_SCUB.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_DP_EXACT_SCUB_4.txt ./data/stackoverflow/label_4.txt 22
#python DP_EXACT_SCUB.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_DP_EXACT_SCUB_5.txt ./data/stackoverflow/label_5.txt 21

#python DP_EXACT_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_DP_EXACT_SCUB_1.txt ./data/wiki/label_1.txt 11
#python DP_EXACT_SCUB.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_DP_EXACT_SCUB_2.txt ./data/wiki/label_2.txt 10
#python DP_EXACT_SCUB.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_DP_EXACT_SCUB_3.txt ./data/wiki/label_3.txt 11
#python DP_EXACT_SCUB.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_DP_EXACT_SCUB_4.txt ./data/wiki/label_4.txt 17
#python DP_EXACT_SCUB.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_DP_EXACT_SCUB_5.txt ./data/wiki/label_5.txt 25

#python DP_EXACT_SCUB.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_DP_EXACT_SCUB_1.txt ./data/epinions/label_1.txt 23
#python DP_EXACT_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_DP_EXACT_SCUB_2.txt ./data/epinions/label_2.txt 23
#python DP_EXACT_SCUB.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_DP_EXACT_SCUB_3.txt ./data/epinions/label_3.txt 23
#python DP_EXACT_SCUB.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_DP_EXACT_SCUB_4.txt ./data/epinions/label_4.txt 23
#python DP_EXACT_SCUB.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_DP_EXACT_SCUB_5.txt ./data/epinions/label_5.txt 23

#python DP_EXACT_SCUB.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_DP_EXACT_SCUB_1.txt ./data/worldcup/label_1.txt 15



#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_1.txt ./data/STACKEXCH/label_1.txt 22
#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_2.txt ./data/STACKEXCH/label_2.txt 22
#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_3.txt ./data/STACKEXCH/label_3.txt 22
#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_4.txt ./data/STACKEXCH/label_4.txt 16
#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_5.txt ./data/STACKEXCH/label_5.txt 22

#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_1.txt ./data/stackoverflow/label_1.txt 22
#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_2.txt ./data/stackoverflow/label_2.txt 22
#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_3.txt ./data/stackoverflow/label_3.txt 22
#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_4.txt ./data/stackoverflow/label_4.txt 22
#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_5.txt ./data/stackoverflow/label_5.txt 21

#python TD_EXACT_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_EXACT_SCUB_1.txt ./data/wiki/label_1.txt 11
#python TD_EXACT_SCUB.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_EXACT_SCUB_2.txt ./data/wiki/label_2.txt 10
#python TD_EXACT_SCUB.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_EXACT_SCUB_3.txt ./data/wiki/label_3.txt 11
#python TD_EXACT_SCUB.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_EXACT_SCUB_4.txt ./data/wiki/label_4.txt 17
#python TD_EXACT_SCUB.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_EXACT_SCUB_5.txt ./data/wiki/label_5.txt 25

#python TD_EXACT_SCUB.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_EXACT_SCUB_1.txt ./data/epinions/label_1.txt 23
#python TD_EXACT_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_EXACT_SCUB_2.txt ./data/epinions/label_2.txt 23
#python TD_EXACT_SCUB.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_EXACT_SCUB_3.txt ./data/epinions/label_3.txt 23
#python TD_EXACT_SCUB.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_EXACT_SCUB_4.txt ./data/epinions/label_4.txt 23
#python TD_EXACT_SCUB.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_EXACT_SCUB_5.txt ./data/epinions/label_5.txt 23

#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_5.txt ./data/fb-friends/label_5.txt 8

#python TD_EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_EXACT_SCUB_1.txt ./data/dblp/label_1.txt 5
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_EXACT_SCUB_2.txt ./data/dblp/label_2.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_EXACT_SCUB_3.txt ./data/dblp/label_3.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_EXACT_SCUB_4.txt ./data/dblp/label_4.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_EXACT_SCUB_5.txt ./data/dblp/label_5.txt 8

#python TD_EXACT_SCUB.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_EXACT_SCUB_1.txt ./data/worldcup/label_1.txt 15








#python DP_G_SCUB.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_DP_G_SCUB_1.txt ./data/epinions/label_1.txt 23
#python DP_G_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_DP_G_SCUB_2.txt ./data/epinions/label_2.txt 23
#python DP_G_SCUB.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_DP_G_SCUB_3.txt ./data/epinions/label_3.txt 23
#python DP_G_SCUB.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_DP_G_SCUB_4.txt ./data/epinions/label_4.txt 23
#python DP_G_SCUB.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_DP_G_SCUB_5.txt ./data/epinions/label_5.txt 23

#python DP_G_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_DP_G_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python DP_G_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_DP_G_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python DP_G_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_DP_G_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python DP_G_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_DP_G_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python DP_G_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_DP_G_SCUB_5.txt ./data/fb-friends/label_5.txt 8

#python DP_G_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_DP_G_SCUB_1.txt ./data/dblp/label_1.txt 5
#python DP_G_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_DP_G_SCUB_2.txt ./data/dblp/label_2.txt 6
#python DP_G_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_DP_G_SCUB_3.txt ./data/dblp/label_3.txt 6
#python DP_G_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_DP_G_SCUB_4.txt ./data/dblp/label_4.txt 6
#python DP_G_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_DP_G_SCUB_5.txt ./data/dblp/label_5.txt 8

#python DP_G_SCUB.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_DP_G_SCUB_1.txt ./data/worldcup/label_1.txt 15

#python DP_G_SCUB.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_DP_G_SCUB_1.txt ./data/STACKEXCH/label_1.txt 22
#python DP_G_SCUB.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_DP_G_SCUB_2.txt ./data/STACKEXCH/label_2.txt 22
#python DP_G_SCUB.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_DP_G_SCUB_3.txt ./data/STACKEXCH/label_3.txt 22
#python DP_G_SCUB.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_DP_G_SCUB_4.txt ./data/STACKEXCH/label_4.txt 16
#python DP_G_SCUB.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_DP_G_SCUB_5.txt ./data/STACKEXCH/label_5.txt 22

#python DP_G_SCUB.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_DP_G_SCUB_1.txt ./data/stackoverflow/label_1.txt 22
#python DP_G_SCUB.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_DP_G_SCUB_2.txt ./data/stackoverflow/label_2.txt 22
#python DP_G_SCUB.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_DP_G_SCUB_3.txt ./data/stackoverflow/label_3.txt 22
#python DP_G_SCUB.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_DP_G_SCUB_4.txt ./data/stackoverflow/label_4.txt 22
#python DP_G_SCUB.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_DP_G_SCUB_5.txt ./data/stackoverflow/label_5.txt 21

#python DP_G_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_DP_G_SCUB_1.txt ./data/wiki/label_1.txt 11
#python DP_G_SCUB.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_DP_G_SCUB_2.txt ./data/wiki/label_2.txt 10
#python DP_G_SCUB.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_DP_G_SCUB_3.txt ./data/wiki/label_3.txt 11
#python DP_G_SCUB.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_DP_G_SCUB_4.txt ./data/wiki/label_4.txt 17
#python DP_G_SCUB.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_DP_G_SCUB_5.txt ./data/wiki/label_5.txt 25




#python TD_G_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_G_SCUB_1.txt ./data/wiki/label_1.txt 11
#python TD_G_SCUB.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_G_SCUB_2.txt ./data/wiki/label_2.txt 10
#python TD_G_SCUB.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_G_SCUB_3.txt ./data/wiki/label_3.txt 11
#python TD_G_SCUB.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_G_SCUB_4.txt ./data/wiki/label_4.txt 17
#python TD_G_SCUB.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_G_SCUB_5.txt ./data/wiki/label_5.txt 25

#python TD_G_SCUB.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_1.txt ./data/stackoverflow/label_1.txt 22
#python TD_G_SCUB.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_2.txt ./data/stackoverflow/label_2.txt 22
#python TD_G_SCUB.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_3.txt ./data/stackoverflow/label_3.txt 22
#python TD_G_SCUB.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_4.txt ./data/stackoverflow/label_4.txt 22
#python TD_G_SCUB.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_5.txt ./data/stackoverflow/label_5.txt 21

#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_1.txt ./data/STACKEXCH/label_1.txt 22
#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_2.txt ./data/STACKEXCH/label_2.txt 22
#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_3.txt ./data/STACKEXCH/label_3.txt 22
#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_4.txt ./data/STACKEXCH/label_4.txt 16
#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_5.txt ./data/STACKEXCH/label_5.txt 22

#python TD_G_SCUB.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_G_SCUB_1.txt ./data/worldcup/label_1.txt 15

#python TD_G_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_G_SCUB_1.txt ./data/dblp/label_1.txt 5
#python TD_G_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_G_SCUB_2.txt ./data/dblp/label_2.txt 6
#python TD_G_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_G_SCUB_3.txt ./data/dblp/label_3.txt 6
#python TD_G_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_G_SCUB_4.txt ./data/dblp/label_4.txt 6
#python TD_G_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_G_SCUB_5.txt ./data/dblp/label_5.txt 8

#python TD_G_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_G_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python TD_G_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_G_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python TD_G_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_G_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python TD_G_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_G_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python TD_G_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_G_SCUB_5.txt ./data/fb-friends/label_5.txt 8

#python TD_G_SCUB.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_G_SCUB_1.txt ./data/epinions/label_1.txt 23
#python TD_G_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_G_SCUB_2.txt ./data/epinions/label_2.txt 23
#python TD_G_SCUB.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_G_SCUB_3.txt ./data/epinions/label_3.txt 23
#python TD_G_SCUB.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_G_SCUB_4.txt ./data/epinions/label_4.txt 23
#python TD_G_SCUB.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_G_SCUB_5.txt ./data/epinions/label_5.txt 23









#python TD_EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_EXACT_SCUB_1.txt ./data/dblp/label_1.txt 5
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_EXACT_SCUB_2.txt ./data/dblp/label_2.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_EXACT_SCUB_3.txt ./data/dblp/label_3.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_EXACT_SCUB_4.txt ./data/dblp/label_4.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_EXACT_SCUB_5.txt ./data/dblp/label_5.txt 8

#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_5.txt ./data/fb-friends/label_5.txt 8


#python DP_G_SCUB.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_DP_G_SCUB_1.txt ./data/epinions/label_1.txt 23
#python DP_G_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_DP_G_SCUB_2.txt ./data/epinions/label_2.txt 23
#python DP_G_SCUB.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_DP_G_SCUB_3.txt ./data/epinions/label_3.txt 23
#python DP_G_SCUB.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_DP_G_SCUB_4.txt ./data/epinions/label_4.txt 23
#python DP_G_SCUB.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_DP_G_SCUB_5.txt ./data/epinions/label_5.txt 23


#python DP_G_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_DP_G_SCUB_1.txt ./data/wiki/label_1.txt 11
#python DP_G_SCUB.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_DP_G_SCUB_2.txt ./data/wiki/label_2.txt 10
#python DP_G_SCUB.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_DP_G_SCUB_3.txt ./data/wiki/label_3.txt 11
#python DP_G_SCUB.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_DP_G_SCUB_4.txt ./data/wiki/label_4.txt 17
#python DP_G_SCUB.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_DP_G_SCUB_5.txt ./data/wiki/label_5.txt 25


#python DP_G_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_DP_G_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python DP_G_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_DP_G_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python DP_G_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_DP_G_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python DP_G_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_DP_G_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python DP_G_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_DP_G_SCUB_5.txt ./data/fb-friends/label_5.txt 8













#python TD_G_SCUB.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_G_SCUB_1.txt ./data/epinions/label_1.txt 23
#python TD_G_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_G_SCUB_2.txt ./data/epinions/label_2.txt 23
#python TD_G_SCUB.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_G_SCUB_3.txt ./data/epinions/label_3.txt 23
#python TD_G_SCUB.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_G_SCUB_4.txt ./data/epinions/label_4.txt 23
#python TD_G_SCUB.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_G_SCUB_5.txt ./data/epinions/label_5.txt 23

#python TD_G_SCUB.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_1.txt ./data/stackoverflow/label_1.txt 22
#python TD_G_SCUB.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_2.txt ./data/stackoverflow/label_2.txt 22
#python TD_G_SCUB.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_3.txt ./data/stackoverflow/label_3.txt 22
#python TD_G_SCUB.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_4.txt ./data/stackoverflow/label_4.txt 22
#python TD_G_SCUB.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_G_SCUB_5.txt ./data/stackoverflow/label_5.txt 21

#python TD_G_SCUB.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_G_SCUB_1.txt ./data/worldcup/label_1.txt 15

#python TD_G_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_G_SCUB_1.txt ./data/wiki/label_1.txt 11
#python TD_G_SCUB.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_G_SCUB_2.txt ./data/wiki/label_2.txt 10
#python TD_G_SCUB.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_G_SCUB_3.txt ./data/wiki/label_3.txt 11
#python TD_G_SCUB.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_G_SCUB_4.txt ./data/wiki/label_4.txt 17
#python TD_G_SCUB.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_G_SCUB_5.txt ./data/wiki/label_5.txt 25

#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_1.txt ./data/STACKEXCH/label_1.txt 22
#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_2.txt ./data/STACKEXCH/label_2.txt 22
#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_3.txt ./data/STACKEXCH/label_3.txt 22
#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_4.txt ./data/STACKEXCH/label_4.txt 16
#python TD_G_SCUB.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_G_SCUB_5.txt ./data/STACKEXCH/label_5.txt 22















#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_1.txt ./data/STACKEXCH/label_1.txt 22
#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_2.txt ./data/STACKEXCH/label_2.txt 22
#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_3.txt ./data/STACKEXCH/label_3.txt 22
#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_4.txt ./data/STACKEXCH/label_4.txt 16
#python TD_EXACT_SCUB.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_EXACT_SCUB_5.txt ./data/STACKEXCH/label_5.txt 22


#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_1.txt ./data/stackoverflow/label_1.txt 22
#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_2.txt ./data/stackoverflow/label_2.txt 22
#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_3.txt ./data/stackoverflow/label_3.txt 22
#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_4.txt ./data/stackoverflow/label_4.txt 22
#python TD_EXACT_SCUB.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_EXACT_SCUB_5.txt ./data/stackoverflow/label_5.txt 21

#python TD_EXACT_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_EXACT_SCUB_1.txt ./data/wiki/label_1.txt 11
#python TD_EXACT_SCUB.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_EXACT_SCUB_2.txt ./data/wiki/label_2.txt 10
#python TD_EXACT_SCUB.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_EXACT_SCUB_3.txt ./data/wiki/label_3.txt 11
#python TD_EXACT_SCUB.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_EXACT_SCUB_4.txt ./data/wiki/label_4.txt 17
#python TD_EXACT_SCUB.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_EXACT_SCUB_5.txt ./data/wiki/label_5.txt 25

#python TD_EXACT_SCUB.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_EXACT_SCUB_1.txt ./data/worldcup/label_1.txt 15

#python TD_EXACT_SCUB.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_EXACT_SCUB_1.txt ./data/epinions/label_1.txt 23
#python TD_EXACT_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_EXACT_SCUB_2.txt ./data/epinions/label_2.txt 23
#python TD_EXACT_SCUB.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_EXACT_SCUB_3.txt ./data/epinions/label_3.txt 23
#python TD_EXACT_SCUB.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_EXACT_SCUB_4.txt ./data/epinions/label_4.txt 23
#python TD_EXACT_SCUB.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_EXACT_SCUB_5.txt ./data/epinions/label_5.txt 23

#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_5.txt ./data/fb-friends/label_5.txt 8

#python TD_EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_EXACT_SCUB_1.txt ./data/dblp/label_1.txt 5
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_EXACT_SCUB_2.txt ./data/dblp/label_2.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_EXACT_SCUB_3.txt ./data/dblp/label_3.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_EXACT_SCUB_4.txt ./data/dblp/label_4.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_EXACT_SCUB_5.txt ./data/dblp/label_5.txt 8











#python TD_G_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_G_SCUB_1.txt ./data/dblp/label_1.txt 5
#python TD_G_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_G_SCUB_2.txt ./data/dblp/label_2.txt 6
#python TD_G_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_G_SCUB_3.txt ./data/dblp/label_3.txt 6
#python TD_G_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_G_SCUB_4.txt ./data/dblp/label_4.txt 6
#python TD_G_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_G_SCUB_5.txt ./data/dblp/label_5.txt 8

#python TD_G_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_G_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python TD_G_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_G_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python TD_G_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_G_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python TD_G_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_G_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python TD_G_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_G_SCUB_5.txt ./data/fb-friends/label_5.txt 8


#python DP_G_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_DP_G_SCUB_1.txt ./data/dblp/label_1.txt 5
#python DP_G_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_DP_G_SCUB_2.txt ./data/dblp/label_2.txt 6
#python DP_G_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_DP_G_SCUB_3.txt ./data/dblp/label_3.txt 6
#python DP_G_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_DP_G_SCUB_4.txt ./data/dblp/label_4.txt 6
#python DP_G_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_DP_G_SCUB_5.txt ./data/dblp/label_5.txt 8



# python TD_HEU.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_HEU_1.txt ./data/dblp/label_1.txt 5
# python TD_HEU.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_HEU_2.txt ./data/dblp/label_2.txt 6
# python TD_HEU.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_HEU_3.txt ./data/dblp/label_3.txt 6
# python TD_HEU.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_HEU_4.txt ./data/dblp/label_4.txt 6
# python TD_HEU.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_HEU_5.txt ./data/dblp/label_5.txt 8

# python TD_HEU.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_HEU_1.txt ./data/fb-friends/label_1.txt 14
# python TD_HEU.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_HEU_2.txt ./data/fb-friends/label_2.txt 11
# python TD_HEU.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_HEU_3.txt ./data/fb-friends/label_3.txt 20
# python TD_HEU.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_HEU_4.txt ./data/fb-friends/label_4.txt 19
# python TD_HEU.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_HEU_5.txt ./data/fb-friends/label_5.txt 8

# python TD_HEU.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_HEU_1.txt ./data/epinions/label_1.txt 23
# python TD_HEU.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_HEU_2.txt ./data/epinions/label_2.txt 23
# python TD_HEU.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_HEU_3.txt ./data/epinions/label_3.txt 23
# python TD_HEU.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_HEU_4.txt ./data/epinions/label_4.txt 23
# python TD_HEU.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_HEU_5.txt ./data/epinions/label_5.txt 23

# python TD_HEU.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_HEU_1.txt ./data/wiki/label_1.txt 11
# python TD_HEU.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_HEU_2.txt ./data/wiki/label_2.txt 10
# python TD_HEU.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_HEU_3.txt ./data/wiki/label_3.txt 11
# python TD_HEU.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_HEU_4.txt ./data/wiki/label_4.txt 17
# python TD_HEU.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_HEU_5.txt ./data/wiki/label_5.txt 25

# python TD_HEU.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_HEU_1.txt ./data/stackoverflow/label_1.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_HEU_2.txt ./data/stackoverflow/label_2.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_HEU_3.txt ./data/stackoverflow/label_3.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_HEU_4.txt ./data/stackoverflow/label_4.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_HEU_5.txt ./data/stackoverflow/label_5.txt 21

# python TD_HEU.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_1.txt ./data/STACKEXCH/label_1.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_2.txt ./data/STACKEXCH/label_2.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_3.txt ./data/STACKEXCH/label_3.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_4.txt ./data/STACKEXCH/label_4.txt 16
# python TD_HEU.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_5.txt ./data/STACKEXCH/label_5.txt 22

#python TD_HEU.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_HEU_1.txt ./data/worldcup/label_1.txt 15


#python TD_HEUS.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_1.txt ./data/stackoverflow/label_1.txt 22
#python TD_HEUS.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_2.txt ./data/stackoverflow/label_2.txt 22
#python TD_HEUS.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_3.txt ./data/stackoverflow/label_3.txt 22
#python TD_HEUS.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_4.txt ./data/stackoverflow/label_4.txt 22
#python TD_HEUS.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_5.txt ./data/stackoverflow/label_5.txt 21

#python TD_HEUS.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_1.txt ./data/STACKEXCH/label_1.txt 22
#python TD_HEUS.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_2.txt ./data/STACKEXCH/label_2.txt 22
#python TD_HEUS.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_3.txt ./data/STACKEXCH/label_3.txt 22
#python TD_HEUS.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_4.txt ./data/STACKEXCH/label_4.txt 16
#python TD_HEUS.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_5.txt ./data/STACKEXCH/label_5.txt 22

#python TD_HEUS.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_HEUS_1.txt ./data/worldcup/label_1.txt 15

# python TD_HEUS.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_HEUS_1.txt ./data/dblp/label_1.txt 5
# python TD_HEUS.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_HEUS_2.txt ./data/dblp/label_2.txt 6
# python TD_HEUS.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_HEUS_3.txt ./data/dblp/label_3.txt 6
# python TD_HEUS.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_HEUS_4.txt ./data/dblp/label_4.txt 6
# python TD_HEUS.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_HEUS_5.txt ./data/dblp/label_5.txt 8

# python TD_HEUS.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_HEUS_1.txt ./data/fb-friends/label_1.txt 14
# python TD_HEUS.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_HEUS_2.txt ./data/fb-friends/label_2.txt 11
# python TD_HEUS.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_HEUS_3.txt ./data/fb-friends/label_3.txt 20
# python TD_HEUS.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_HEUS_4.txt ./data/fb-friends/label_4.txt 19
# python TD_HEUS.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_HEUS_5.txt ./data/fb-friends/label_5.txt 8

# python TD_HEUS.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_HEUS_1.txt ./data/epinions/label_1.txt 23
# python TD_HEUS.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_HEUS_2.txt ./data/epinions/label_2.txt 23
# python TD_HEUS.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_HEUS_3.txt ./data/epinions/label_3.txt 23
# python TD_HEUS.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_HEUS_4.txt ./data/epinions/label_4.txt 23
# python TD_HEUS.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_HEUS_5.txt ./data/epinions/label_5.txt 23

# python TD_HEUS.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_HEUS_1.txt ./data/wiki/label_1.txt 11
# python TD_HEUS.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_HEUS_2.txt ./data/wiki/label_2.txt 10
# python TD_HEUS.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_HEUS_3.txt ./data/wiki/label_3.txt 11
# python TD_HEUS.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_HEUS_4.txt ./data/wiki/label_4.txt 17
# python TD_HEUS.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_HEUS_5.txt ./data/wiki/label_5.txt 25

#python TD_HEU.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_HEU_1.txt ./data/stackoverflow/label_1.txt 22
#python TD_HEU.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_HEU_2.txt ./data/stackoverflow/label_2.txt 22
#python TD_HEU.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_HEU_3.txt ./data/stackoverflow/label_3.txt 22
#python TD_HEU.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_HEU_4.txt ./data/stackoverflow/label_4.txt 22
#python TD_HEU.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_HEU_5.txt ./data/stackoverflow/label_5.txt 21

#python TD_HEU.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_1.txt ./data/STACKEXCH/label_1.txt 22
#python TD_HEU.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_2.txt ./data/STACKEXCH/label_2.txt 22
#python TD_HEU.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_3.txt ./data/STACKEXCH/label_3.txt 22
#python TD_HEU.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_4.txt ./data/STACKEXCH/label_4.txt 16
#python TD_HEU.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_5.txt ./data/STACKEXCH/label_5.txt 22

#python TD_HEUS.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_1.txt ./data/stackoverflow/label_1.txt 22
#python TD_HEUS.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_2.txt ./data/stackoverflow/label_2.txt 22
#python TD_HEUS.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_3.txt ./data/stackoverflow/label_3.txt 22
#python TD_HEUS.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_4.txt ./data/stackoverflow/label_4.txt 22
#python TD_HEUS.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_5.txt ./data/stackoverflow/label_5.txt 21

#python TD_HEUS.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_1.txt ./data/STACKEXCH/label_1.txt 22
#python TD_HEUS.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_2.txt ./data/STACKEXCH/label_2.txt 22
#python TD_HEUS.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_3.txt ./data/STACKEXCH/label_3.txt 22
#python TD_HEUS.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_4.txt ./data/STACKEXCH/label_4.txt 16
#python TD_HEUS.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_5.txt ./data/STACKEXCH/label_5.txt 22

#python get_JS_runtime_all_methods_vary_JD.py ./data/dblp/ego_list_1.pkl 1,2,4,7,8,10
#python EXACT_SCUB_vs_GSCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_EXACT_SCUB_vs_GSCUB.txt 25

# python TD_HEUS.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_HEUS_1.txt ./data/dblp/label_1.txt 5
# python TD_HEU.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_HEU_2.txt ./data/dblp/label_2.txt 6
# python TD_HEU.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_HEU_3.txt ./data/dblp/label_3.txt 6
# python TD_HEU.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_HEU_4.txt ./data/dblp/label_4.txt 6
# python TD_HEU.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_HEU_5.txt ./data/dblp/label_5.txt 8

# python TD_HEU.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_HEU_1.txt ./data/fb-friends/label_1.txt 14
# python TD_HEU.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_HEU_2.txt ./data/fb-friends/label_2.txt 11
# python TD_HEU.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_HEU_3.txt ./data/fb-friends/label_3.txt 20
# python TD_HEU.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_HEU_4.txt ./data/fb-friends/label_4.txt 19
# python TD_HEU.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_HEU_5.txt ./data/fb-friends/label_5.txt 8

# python TD_HEU.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_HEU_1.txt ./data/epinions/label_1.txt 23
# python TD_HEU.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_HEU_2.txt ./data/epinions/label_2.txt 23
# python TD_HEU.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_HEU_3.txt ./data/epinions/label_3.txt 23
# python TD_HEU.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_HEU_4.txt ./data/epinions/label_4.txt 23
# python TD_HEU.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_HEU_5.txt ./data/epinions/label_5.txt 23

# python TD_HEU.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_HEU_1.txt ./data/wiki/label_1.txt 11
# python TD_HEU.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_HEU_2.txt ./data/wiki/label_2.txt 10
# python TD_HEU.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_HEU_3.txt ./data/wiki/label_3.txt 11
# python TD_HEU.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_HEU_4.txt ./data/wiki/label_4.txt 17
# python TD_HEU.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_HEU_5.txt ./data/wiki/label_5.txt 25

# python TD_HEUS.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_1.txt ./data/stackoverflow/label_1.txt 22
# python TD_HEUS.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_2.txt ./data/stackoverflow/label_2.txt 22
# python TD_HEUS.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_3.txt ./data/stackoverflow/label_3.txt 22
# python TD_HEUS.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_4.txt ./data/stackoverflow/label_4.txt 22
# python TD_HEUS.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_HEUS_5.txt ./data/stackoverflow/label_5.txt 21

# python TD_HEU.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_HEU_1.txt ./data/stackoverflow/label_1.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_HEU_2.txt ./data/stackoverflow/label_2.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_HEU_3.txt ./data/stackoverflow/label_3.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_HEU_4.txt ./data/stackoverflow/label_4.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_HEU_5.txt ./data/stackoverflow/label_5.txt 21

# python TD_HEUS.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_1.txt ./data/STACKEXCH/label_1.txt 22
# python TD_HEUS.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_2.txt ./data/STACKEXCH/label_2.txt 22
# python TD_HEUS.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_3.txt ./data/STACKEXCH/label_3.txt 22
# python TD_HEUS.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_4.txt ./data/STACKEXCH/label_4.txt 16
# python TD_HEUS.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_HEUS_5.txt ./data/STACKEXCH/label_5.txt 22

# python TD_HEU.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_1.txt ./data/STACKEXCH/label_1.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_2.txt ./data/STACKEXCH/label_2.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_3.txt ./data/STACKEXCH/label_3.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_4.txt ./data/STACKEXCH/label_4.txt 16
# python TD_HEU.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_5.txt ./data/STACKEXCH/label_5.txt 22

#python TD_HEUS.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_HEUS_1.txt ./data/worldcup/label_1.txt 15
#python TD_HEU.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_HEU_1.txt ./data/worldcup/label_1.txt 15


#python TD_HEU.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_HEU_1.txt ./data/dblp/label_1.txt 5
#python TD_HEU.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_HEU_2.txt ./data/dblp/label_2.txt 6
#python TD_HEU.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_HEU_3.txt ./data/dblp/label_3.txt 6
#python TD_HEU.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_HEU_4.txt ./data/dblp/label_4.txt 6
#python TD_HEU.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_HEU_5.txt ./data/dblp/label_5.txt 8

#python TD_HEU.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_HEU_1.txt ./data/fb-friends/label_1.txt 14
#python TD_HEU.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_HEU_2.txt ./data/fb-friends/label_2.txt 11
#python TD_HEU.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_HEU_3.txt ./data/fb-friends/label_3.txt 20
#python TD_HEU.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_HEU_4.txt ./data/fb-friends/label_4.txt 19
#python TD_HEU.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_HEU_5.txt ./data/fb-friends/label_5.txt 8

# python TD_HEU.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_HEU_1.txt ./data/epinions/label_1.txt 23
# python TD_HEU.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_HEU_2.txt ./data/epinions/label_2.txt 23
# python TD_HEU.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_HEU_3.txt ./data/epinions/label_3.txt 23
# python TD_HEU.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_HEU_4.txt ./data/epinions/label_4.txt 23
# python TD_HEU.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_HEU_5.txt ./data/epinions/label_5.txt 23

# python TD_HEU.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_HEU_1.txt ./data/wiki/label_1.txt 11
# python TD_HEU.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_HEU_2.txt ./data/wiki/label_2.txt 10
# python TD_HEU.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_HEU_3.txt ./data/wiki/label_3.txt 11
# python TD_HEU.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_HEU_4.txt ./data/wiki/label_4.txt 17
# python TD_HEU.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_HEU_5.txt ./data/wiki/label_5.txt 25

# python TD_HEU.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_HEU_1.txt ./data/stackoverflow/label_1.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_HEU_2.txt ./data/stackoverflow/label_2.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_HEU_3.txt ./data/stackoverflow/label_3.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_HEU_4.txt ./data/stackoverflow/label_4.txt 22
# python TD_HEU.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_HEU_5.txt ./data/stackoverflow/label_5.txt 21

# python TD_HEU.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_1.txt ./data/STACKEXCH/label_1.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_2.txt ./data/STACKEXCH/label_2.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_3.txt ./data/STACKEXCH/label_3.txt 22
# python TD_HEU.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_4.txt ./data/STACKEXCH/label_4.txt 16
# python TD_HEU.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_5.txt ./data/STACKEXCH/label_5.txt 22

# python TD_HEU.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_HEU_1.txt ./data/worldcup/label_1.txt 15



#python DP_EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_DP_EXACT_SCUB_1.txt ./data/dblp/label_1.txt 5
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_DP_EXACT_SCUB_2.txt ./data/dblp/label_2.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_DP_EXACT_SCUB_3.txt ./data/dblp/label_3.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_DP_EXACT_SCUB_4.txt ./data/dblp/label_4.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_DP_EXACT_SCUB_5.txt ./data/dblp/label_5.txt 8


#python TD_EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_EXACT_SCUB_1.txt ./data/dblp/label_1.txt 5
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_EXACT_SCUB_2.txt ./data/dblp/label_2.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_EXACT_SCUB_3.txt ./data/dblp/label_3.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_EXACT_SCUB_4.txt ./data/dblp/label_4.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_EXACT_SCUB_5.txt ./data/dblp/label_5.txt 8

#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_5.txt ./data/fb-friends/label_5.txt 8


#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_heuristicswithP3_1.txt 0.1,0.2,0.3,0.5,0.7,1 10 5
#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_heuristicswithP3_2.txt 0.1 5,10,15,20 5

#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_heuristicswithP3_1.txt 0.2,0.5,0.7,1 40 60
#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_heuristicswithP3_2.txt 0.2 40,50,60,70 60

#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_heuristicswithP3_1.txt 0.1,0.2,0.5,0.7,1 25 60
#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_heuristicswithP3_2.txt 0.5 10,15,20,25,30,50,70 60

#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_heuristicswithP3_1.txt 0.1,0.2,0.3,0.5,0.7,1 20 70
#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_heuristicswithP3_2.txt 0.2 10,20,30,40,50,60,70 70

#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_heuristicswithP3_1.txt 0.1,0.2,0.5,0.7,1 60 70
#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_heuristicswithP3_2.txt 1 10,20,30,40,50,60,70 70

#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_heuristicswithP3_1.txt 0.1,0.2,0.5,0.7,1 30 70
#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_heuristicswithP3_2.txt 1 10,20,30,40,50,60,70 70

#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_heuristicswithP3_1.txt 0.02,0.03,0.1,0.2,0.7,1 20 70
#python get_JS_runtime_all_methods_heuristicswithP3.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_heuristicswithP3_2.txt 0.02 10,20,30,60,90,120,150 70

#python EXACT_SCUB_DP.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_EXACT_SCUB_DP_1.txt ./data/dblp/label_1.txt 5
#python EXACT_SCUB_DP.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_EXACT_SCUB_DP_2.txt ./data/dblp/label_2.txt 6
#python EXACT_SCUB_DP.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_EXACT_SCUB_DP_3.txt ./data/dblp/label_3.txt 6
#python EXACT_SCUB_DP.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_EXACT_SCUB_DP_4.txt ./data/dblp/label_4.txt 6
#python EXACT_SCUB_DP.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_EXACT_SCUB_DP_5.txt ./data/dblp/label_5.txt 8

#python EXACT_SCUB_TD.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_EXACT_SCUB_TD_1.txt ./data/dblp/label_1.txt 5
#python EXACT_SCUB_TD.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_EXACT_SCUB_TD_2.txt ./data/dblp/label_2.txt 6
#python EXACT_SCUB_TD.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_EXACT_SCUB_TD_3.txt ./data/dblp/label_3.txt 6
#python EXACT_SCUB_TD.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_EXACT_SCUB_TD_4.txt ./data/dblp/label_4.txt 6
#python EXACT_SCUB_TD.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_EXACT_SCUB_TD_5.txt ./data/dblp/label_5.txt 8

#python EXACT_SCUB_TD.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_EXACT_SCUB_TD_1.txt ./data/fb-friends/label_1.txt 14
#python EXACT_SCUB_TD.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_EXACT_SCUB_TD_2.txt ./data/fb-friends/label_2.txt 11
#python EXACT_SCUB_TD.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_EXACT_SCUB_TD_3.txt ./data/fb-friends/label_3.txt 20
#python EXACT_SCUB_TD.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_EXACT_SCUB_TD_4.txt ./data/fb-friends/label_4.txt 19
#python EXACT_SCUB_TD.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_EXACT_SCUB_TD_5.txt ./data/fb-friends/label_5.txt 8

#python HUE_for_WOC.py ./data/worldcup/WOC.pkl ./data/worldcup/WOC.txt 3 1 5 30 false
#python faster_heuristics_WOC.py ./data/worldcup/WOC.pkl ./data/worldcup/WOC.txt 3

#python DP_Greedy_DBLP.py ./data/worldcup/WOC.pkl ./data/worldcup/WOC.txt 3

#python DP_Greedy_DBLP.py ./data/dblp/Phillip_Yu.pkl ./data/dblp/Phillip_Yu.txt 4
#python DP_Greedy_DBLP.py ./data/dblp/Phillip_Yu.pkl ./data/dblp/Phillip_Yu.txt 4

#python EXACT_SCUB_vs_GSCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_EXACT_SCUB_vs_GSCUB.txt 25

#python EXACT_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_EXACT_SCUB_1.txt 0.1,0.2,0.3,0.5,0.7,1 20 70
#python EXACT_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_EXACT_SCUB_2.txt 0.2 10,20,30,40,50,60,70 70

#python EXACT_SCUB_vs_GSCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_EXACT_SCUB_vs_GSCUB.txt 5



#python ILP_JD_UB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_ILP_JD_UB_1.txt 0.5 10 5

#python get_JS_runtime_all_methods.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_all_methods_1.txt 0.1,0.2,0.3,0.5,0.7,1 10 3 5
#python get_JS_runtime_all_methods.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_all_methods_2.txt 0.1 5,10,15,20 3 5

#python get_JS_runtime_all_methods.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_all_methods_1.txt 0.2,0.5,0.7,1 40 5 60
#python get_JS_runtime_all_methods.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_all_methods_2.txt 0.2 40,50,60,70 5 60

#python EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_EXACT_SCUB_1.txt 0.1,0.2,0.3,0.5,0.7,1 10 5
#python EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_EXACT_SCUB_2.txt 0.1 5,10,15,20 5

#python EXACT_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_EXACT_SCUB_1.txt 0.2,0.5,0.7,1 40 60
#python EXACT_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_EXACT_SCUB_2.txt 0.2 40,50,60,70 60

#python EXACT_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_EXACT_SCUB_1.txt 0.1,0.2,0.5,0.7,1 25 60
#python EXACT_SCUB.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_EXACT_SCUB_2.txt 0.5 10,15,20,25,30,50,70 60

#python EXACT_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_EXACT_SCUB_1.txt 0.1,0.2,0.3,0.5,0.7,1 20 70
#python EXACT_SCUB.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_EXACT_SCUB_2.txt 0.2 10,20,30,40,50,60,70 70

#python EXACT_SCUB.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_EXACT_SCUB_1.txt 0.1,0.2,0.5,0.7,1 60 70
#python EXACT_SCUB.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_EXACT_SCUB_2.txt 1 10,20,30,40,50,60,70 70

#python EXACT_SCUB.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_EXACT_SCUB_1.txt 0.1,0.2,0.5,0.7,1 30 70
#python EXACT_SCUB.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_EXACT_SCUB_2.txt 1 10,20,30,40,50,60,70 70

#python EXACT_SCUB.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_EXACT_SCUB_1.txt 0.02,0.03,0.1,0.2,0.7,1 20 70
#python EXACT_SCUB.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_EXACT_SCUB_2.txt 0.02 10,20,30,60,90,120,150 70

#python EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_EXACT_SCUB_1.txt 0.1,0.2,0.3,0.5,0.7,1 10 5
#python ILP_JD_UB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_ILP_JD_UB_1.txt 1 10 5

#python get_JS_runtime_all_methods.py ./data/epinions/ego_list_2.pkl ./data/epinions/all_15.txt 0.5 15 5 60

#python get_JS_runtime_ILP.py ./data/epinions/ego_list_2.pkl ./data/epinions/ILP_15.txt 0.5 15 60


#python 2_approximation_TD.py ./data/dblp/ego_list_1.pkl ./data/dblp/2_approxi.txt ./data/dblp/label_1.txt 5
#python faster_heuristics_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_TD_200.txt ./data/worldcup/label_1.txt 5 200
#python faster_heuristics_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_TD_400.txt ./data/worldcup/label_1.txt 5 400
#python faster_heuristics_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_TD_600.txt ./data/worldcup/label_1.txt 5 600
#python faster_heuristics_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_TD_720.txt ./data/worldcup/label_1.txt 5 720

#python faster_heuristics_with_HyperMinHash_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_S_TD_200.txt ./data/worldcup/label_1.txt 5 1 5 30 false 200
#python faster_heuristics_with_HyperMinHash_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_S_TD_400.txt ./data/worldcup/label_1.txt 5 1 5 30 false 400
#python faster_heuristics_with_HyperMinHash_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_S_TD_600.txt ./data/worldcup/label_1.txt 5 1 5 30 false 600
#python faster_heuristics_with_HyperMinHash_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_S_TD_720.txt ./data/worldcup/label_1.txt 5 1 5 30 false 720

#python r_approximation_TD_nophase3.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_old_r_appro_nophase3_TD_1.txt ./data/dblp/label_1.txt 5 5
#python r_approximation_TD_nophase3.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_old_r_appro_nophase3_TD_2.txt ./data/dblp/label_2.txt 6 5
#python r_approximation_TD_nophase3.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_old_r_appro_nophase3_TD_3.txt ./data/dblp/label_3.txt 6 5
#python r_approximation_TD_nophase3.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_old_r_appro_nophase3_TD_4.txt ./data/dblp/label_4.txt 6 5
#python r_approximation_TD_nophase3.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_old_r_appro_nophase3_TD_5.txt ./data/dblp/label_5.txt 8 5

#python r_approximation_TD_new.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_new_r_appro_TD_1.txt ./data/fb-friends/label_1.txt 14 5
#python r_approximation_TD_new.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_new_r_appro_TD_2.txt ./data/fb-friends/label_2.txt 11 5
#python r_approximation_TD_new.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_new_r_appro_TD_3.txt ./data/fb-friends/label_3.txt 20 5
#python r_approximation_TD_new.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_new_r_appro_TD_4.txt ./data/fb-friends/label_4.txt 19 5
#python r_approximation_TD_new.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_new_r_appro_TD_5.txt ./data/fb-friends/label_5.txt 8 5

#python r_approximation_TD_new.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_new_r_appro_TD_1.txt ./data/epinions/label_1.txt 23 5 
#python r_approximation_TD_new.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_new_r_appro_TD_2.txt ./data/epinions/label_2.txt 23 5
#python r_approximation_TD_new.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_new_r_appro_TD_3.txt ./data/epinions/label_3.txt 23 5
#python r_approximation_TD_new.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_new_r_appro_TD_4.txt ./data/epinions/label_4.txt 23 5
#python r_approximation_TD_new.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_new_r_appro_TD_5.txt ./data/epinions/label_5.txt 23 5

#python r_approximation_TD_new.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_new_r_appro_TD_1.txt ./data/wiki/label_1.txt 11 5
#python r_approximation_TD_new.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_new_r_appro_TD_2.txt ./data/wiki/label_2.txt 10 5
#python r_approximation_TD_new.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_new_r_appro_TD_3.txt ./data/wiki/label_3.txt 11 5
#python r_approximation_TD_new.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_new_r_appro_TD_4.txt ./data/wiki/label_4.txt 17 5
#python r_approximation_TD_new.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_new_r_appro_TD_5.txt ./data/wiki/label_5.txt 25 5

#python r_approximation_TD_new.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_new_r_appro_TD_1.txt ./data/stackoverflow/label_1.txt 22 5
#python r_approximation_TD_new.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_new_r_appro_TD_2.txt ./data/stackoverflow/label_2.txt 22 5
#python r_approximation_TD_new.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_new_r_appro_TD_3.txt ./data/stackoverflow/label_3.txt 22 5
#python r_approximation_TD_new.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_new_r_appro_TD_4.txt ./data/stackoverflow/label_4.txt 22 5
#python r_approximation_TD_new.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_new_r_appro_TD_5.txt ./data/stackoverflow/label_5.txt 21 5

#python r_approximation_TD_new.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_new_r_appro_TD_1.txt ./data/STACKEXCH/label_1.txt 22 5
#python r_approximation_TD_new.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_new_r_appro_TD_2.txt ./data/STACKEXCH/label_2.txt 22 5
#python r_approximation_TD_new.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_new_r_appro_TD_3.txt ./data/STACKEXCH/label_3.txt 22 5
#python r_approximation_TD_new.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_new_r_appro_TD_4.txt ./data/STACKEXCH/label_4.txt 16 5
#python r_approximation_TD_new.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_new_r_appro_TD_5.txt ./data/STACKEXCH/label_5.txt 22 5

#python r_approximation_TD_new_nophase3.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_new_r_appro_TD_1.txt ./data/worldcup/label_1.txt 15 5


#python heuristics_with_HyperMinHash_sumJD.py ./data/dblp/ego_list_1.pkl 5 1 5 40 false
#python heuristics_with_HyperMinHash_sumJD.py ./data/fb-friends/ego_list_1.pkl 5 1 5 40 false
#python heuristics_with_HyperMinHash_sumJD.py ./data/epinions/ego_list_1.pkl 5 1 5 40 false
#python heuristics_with_HyperMinHash_sumJD.py ./data/wiki/ego_list_1.pkl 5 1 5 40 false
#python heuristics_with_HyperMinHash_sumJD.py ./data/stackoverflow/ego_list_1.pkl 5 1 5 40 false
#python heuristics_with_HyperMinHash_sumJD.py ./data/STACKEXCH/ego_list_1.pkl 5 1 5 40 false
#python heuristics_with_HyperMinHash_sumJD.py ./data/worldcup/ego_list_1.pkl 5 1 5 40 false

#python TD_HEU.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_HEU_1.txt ./data/dblp/label_1.txt 5
#python TD_HEU.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_HEU_2.txt ./data/dblp/label_2.txt 6
#python TD_HEU.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_HEU_3.txt ./data/dblp/label_3.txt 6
#python TD_HEU.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_HEU_4.txt ./data/dblp/label_4.txt 6
#python TD_HEU.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_HEU_5.txt ./data/dblp/label_5.txt 8

#python TD_HEU.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_HEU_1.txt ./data/fb-friends/label_1.txt 14
#python TD_HEU.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_HEU_2.txt ./data/fb-friends/label_2.txt 11
#python TD_HEU.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_HEU_3.txt ./data/fb-friends/label_3.txt 20
#python TD_HEU.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_HEU_4.txt ./data/fb-friends/label_4.txt 19
#python TD_HEU.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_HEU_5.txt ./data/fb-friends/label_5.txt 8

#python TD_HEU.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_HEU_1.txt ./data/epinions/label_1.txt 23
#python TD_HEU.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_HEU_2.txt ./data/epinions/label_2.txt 23
#python TD_HEU.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_HEU_3.txt ./data/epinions/label_3.txt 23
#python TD_HEU.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_HEU_4.txt ./data/epinions/label_4.txt 23
#python TD_HEU.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_HEU_5.txt ./data/epinions/label_5.txt 23

#python TD_HEU.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_HEU_1.txt ./data/wiki/label_1.txt 11
#python TD_HEU.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_HEU_2.txt ./data/wiki/label_2.txt 10
#python TD_HEU.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_HEU_3.txt ./data/wiki/label_3.txt 11
#python TD_HEU.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_HEU_4.txt ./data/wiki/label_4.txt 17
#python TD_HEU.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_HEU_5.txt ./data/wiki/label_5.txt 25

#python TD_HEU.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_HEU_1.txt ./data/stackoverflow/label_1.txt 22
#python TD_HEU.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_HEU_2.txt ./data/stackoverflow/label_2.txt 22
#python TD_HEU.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_HEU_3.txt ./data/stackoverflow/label_3.txt 22
#python TD_HEU.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_HEU_4.txt ./data/stackoverflow/label_4.txt 22
#python TD_HEU.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_HEU_5.txt ./data/stackoverflow/label_5.txt 21

#python TD_HEU.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_1.txt ./data/STACKEXCH/label_1.txt 22
#python TD_HEU.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_2.txt ./data/STACKEXCH/label_2.txt 22
#python TD_HEU.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_3.txt ./data/STACKEXCH/label_3.txt 22
#python TD_HEU.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_4.txt ./data/STACKEXCH/label_4.txt 16
#python TD_HEU.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_HEU_5.txt ./data/STACKEXCH/label_5.txt 22

#python TD_HEU.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_HEU_1.txt ./data/worldcup/label_1.txt 15

#python DP_Greedy.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_greedy_1.txt ./data/dblp/label_1.txt 5
#python DP_Greedy.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_greedy_2.txt ./data/dblp/label_2.txt 6
#python DP_Greedy.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_greedy_3.txt ./data/dblp/label_3.txt 6
#python DP_Greedy.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_greedy_4.txt ./data/dblp/label_4.txt 6
#python DP_Greedy.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_greedy_5.txt ./data/dblp/label_5.txt 8

#python DP_Greedy.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_greedy_1.txt ./data/fb-friends/label_1.txt 14
#python DP_Greedy.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_greedy_2.txt ./data/fb-friends/label_2.txt 11
#python DP_Greedy.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_greedy_3.txt ./data/fb-friends/label_3.txt 20
#python DP_Greedy.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_greedy_4.txt ./data/fb-friends/label_4.txt 19
#python DP_Greedy.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_greedy_5.txt ./data/fb-friends/label_5.txt 8

#python DP_Greedy.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_greedy_1.txt ./data/epinions/label_1.txt 23
#python DP_Greedy.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_greedy_2.txt ./data/epinions/label_2.txt 23
#python DP_Greedy.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_greedy_3.txt ./data/epinions/label_3.txt 23
#python DP_Greedy.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_greedy_4.txt ./data/epinions/label_4.txt 23
#python DP_Greedy.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_greedy_5.txt ./data/epinions/label_5.txt 23

#python DP_Greedy.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_greedy_1.txt ./data/wiki/label_1.txt 11
#python DP_Greedy.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_greedy_2.txt ./data/wiki/label_2.txt 10
#python DP_Greedy.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_greedy_3.txt ./data/wiki/label_3.txt 11
#python DP_Greedy.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_greedy_4.txt ./data/wiki/label_4.txt 17
#python DP_Greedy.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_greedy_5.txt ./data/wiki/label_5.txt 25

#python DP_Greedy.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_greedy_1.txt ./data/stackoverflow/label_1.txt 22
#python DP_Greedy.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_greedy_2.txt ./data/stackoverflow/label_2.txt 22
#python DP_Greedy.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_greedy_3.txt ./data/stackoverflow/label_3.txt 22
#python DP_Greedy.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_greedy_4.txt ./data/stackoverflow/label_4.txt 22
#python DP_Greedy.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_greedy_5.txt ./data/stackoverflow/label_5.txt 21

#python DP_Greedy.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_greedy_1.txt ./data/STACKEXCH/label_1.txt 22
#python DP_Greedy.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_greedy_2.txt ./data/STACKEXCH/label_2.txt 22
#python DP_Greedy.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_greedy_3.txt ./data/STACKEXCH/label_3.txt 22
#python DP_Greedy.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_greedy_4.txt ./data/STACKEXCH/label_4.txt 16
#python DP_Greedy.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_greedy_5.txt ./data/STACKEXCH/label_5.txt 22

#python DP_Greedy.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_greedy_1.txt ./data/worldcup/label_1.txt 15

#python top_down_greedy.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_greedy_1.txt ./data/dblp/label_1.txt 5
#python top_down_greedy.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_greedy_2.txt ./data/dblp/label_2.txt 6
#python top_down_greedy.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_greedy_3.txt ./data/dblp/label_3.txt 6
#python top_down_greedy.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_greedy_4.txt ./data/dblp/label_4.txt 6
#python top_down_greedy.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_greedy_5.txt ./data/dblp/label_5.txt 8


#python top_down_greedy.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_greedy_1.txt ./data/fb-friends/label_1.txt 14
#python top_down_greedy.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_greedy_2.txt ./data/fb-friends/label_2.txt 11
#python top_down_greedy.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_greedy_3.txt ./data/fb-friends/label_3.txt 20
#python top_down_greedy.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_greedy_4.txt ./data/fb-friends/label_4.txt 19
#python top_down_greedy.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_greedy_5.txt ./data/fb-friends/label_5.txt 8

#python top_down_greedy.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_greedy_1.txt ./data/epinions/label_1.txt 23
#python top_down_greedy.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_greedy_2.txt ./data/epinions/label_2.txt 23
#python top_down_greedy.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_greedy_3.txt ./data/epinions/label_3.txt 23
#python top_down_greedy.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_greedy_4.txt ./data/epinions/label_4.txt 23
#python top_down_greedy.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_greedy_5.txt ./data/epinions/label_5.txt 23

#python top_down_greedy.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_greedy_1.txt ./data/wiki/label_1.txt 11
#python top_down_greedy.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_greedy_2.txt ./data/wiki/label_2.txt 10
#python top_down_greedy.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_greedy_3.txt ./data/wiki/label_3.txt 11
#python top_down_greedy.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_greedy_4.txt ./data/wiki/label_4.txt 17
#python top_down_greedy.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_greedy_5.txt ./data/wiki/label_5.txt 25

#python top_down_greedy.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_greedy_1.txt ./data/stackoverflow/label_1.txt 22
#python top_down_greedy.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_greedy_2.txt ./data/stackoverflow/label_2.txt 22
#python top_down_greedy.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_greedy_3.txt ./data/stackoverflow/label_3.txt 22
#python top_down_greedy.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_greedy_4.txt ./data/stackoverflow/label_4.txt 22
#python top_down_greedy.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_greedy_5.txt ./data/stackoverflow/label_5.txt 21

#python top_down_greedy.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_1.txt ./data/STACKEXCH/label_1.txt 22
#python top_down_greedy.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_2.txt ./data/STACKEXCH/label_2.txt 22
#python top_down_greedy.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_3.txt ./data/STACKEXCH/label_3.txt 22
#python top_down_greedy.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_4.txt ./data/STACKEXCH/label_4.txt 16
#python top_down_greedy.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_5.txt ./data/STACKEXCH/label_5.txt 22

#python top_down_greedy.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_greedy_1.txt ./data/worldcup/label_1.txt 15



#python r_approximation_TD.py ./data/dblp/ego_list_2.pkl ./data/dblp/r_appro_TD_fraction_2.txt ./data/dblp/label_2.txt 6 5

#50/0 40/0.111  25/0.333  15/0.538   8/0.724
#python get_JS_runtime_all_methods_vary_JD.py ./data/dblp/ego_list_1.pkl ./data/dblp/get_JS_runtime_all_methods_4.txt 5 8
#python get_JS_runtime_ILP_2.py ./data/dblp/ego_list_1.pkl ./data/dblp/get_JS_runtime_ILP_2.txt 20
#python get_JS_runtime_ILP.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_ILP_1.txt 0.5 10 5
#python get_JS_runtime_ILP.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_ILP_1.txt 0.1,0.2 10 5
#python get_JS_runtime_ILP.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_ILP_2.txt 0.1 5,10 5

#python get_JS_runtime_ILP.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_ILP_1.txt 0.2 70 60
#python get_JS_runtime_ILP.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_ILP_1.txt 0.1,0.2,0.5,0.7,1 40 60
#python get_JS_runtime_ILP.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_ILP_2.txt 0.2 10,20,30,40 60

#python get_JS_runtime_ILP.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_ILP_1.txt 0.5 70 60
#python get_JS_runtime_ILP.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_ILP_1.txt 0.1,0.2,0.5,0.7,1 25 60
#python get_JS_runtime_ILP.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_ILP_2.txt 0.5 10,20,25 60

#python get_JS_runtime_ILP.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_ILP_1.txt 1 70 70
#python get_JS_runtime_ILP.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_ILP_1.txt 0.1,0.2,0.3 20 70
#python get_JS_runtime_ILP.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_ILP_2.txt 0.2 10,20,30 70

#python get_JS_runtime_ILP.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_ILP_1.txt 1 70 70
#python get_JS_runtime_ILP.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_ILP_1.txt 0.1,0.2,0.5,0.7,1 60 70
#python get_JS_runtime_ILP.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_ILP_2.txt 1 10,20,30,40,50,60 70

#python get_JS_runtime_ILP.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_ILP_1.txt 1 70 70
#python get_JS_runtime_ILP.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_ILP_1.txt 0.1,0.2,0.5,0.7,1 30 70
#python get_JS_runtime_ILP.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_ILP_2.txt 1 10,20,30 70

#python get_JS_runtime_ILP.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_ILP_1.txt 0.02 70 70
#python get_JS_runtime_ILP.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_ILP_1.txt 0.01,0.02,0.03 20 70
#python get_JS_runtime_ILP.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_ILP_2.txt 0.02 5,10,15,20 70


#——————————————————————


#python get_JS_runtime_all_methods_new_r.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_test_new_r_approximation_1.txt 0.1,0.2,0.3,0.5,0.7,1 10 3 5
#python get_JS_runtime_all_methods_new_r.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_test_new_r_approximation_2.txt 0.1 5,10,15,20 3 5
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_test_new_r_approximation_nophase3_1.txt 0.1,0.2,0.3,0.5,0.7,1 10 3 5
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_test_new_r_approximation_nophase3_2.txt 0.1 5,10,15,20 3 5
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_test_old_r_approximation_nophase3_1.txt 0.1,0.2,0.3,0.5,0.7,1 10 3 5
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_test_old_r_approximation_nophase3_2.txt 0.1 5,10,15,20 3 5
#python get_JS_runtime_all_methods.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_all_methods_1.txt 0.1,0.2,0.3,0.5,0.7,1 10 3 5
#python get_JS_runtime_all_methods.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_all_methods_2.txt 0.1 5,10,15,20 3 5


#python get_JS_runtime_all_methods_new_r.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_test_new_r_approximation_1.txt 0.1,0.2,0.5,0.7,1 40 5 60
#python get_JS_runtime_all_methods_new_r.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_test_new_r_approximation_2.txt 0.2 10,20,30,40,50,60,70 5 60
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_test_new_r_approximation_nophase3_1.txt 0.1,0.2,0.5,0.7,1 40 5 60
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_test_new_r_approximation_nophase3_2.txt 0.2 10,20,30,40,50,60,70 5 60
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_test_old_r_approximation_nophase3_1.txt 0.1,0.2,0.5,0.7,1 40 5 60
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_test_old_r_approximation_nophase3_2.txt 0.2 10,20,30,40,50,60,70 5 60
#python get_JS_runtime_all_methods.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_all_methods_1.txt 0.1,0.2,0.5,0.7,1 40 5 60
#python get_JS_runtime_all_methods.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_all_methods_2.txt 0.2 10,20,30,40,50,60,70 5 60


#python get_JS_runtime_all_methods_new_r.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_test_new_r_approximation_1.txt 0.1,0.2,0.5,0.7,1 25 5 60
#python get_JS_runtime_all_methods_new_r.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_test_new_r_approximation_2.txt 0.5 10,20,25,30,40,50,60,70 5 60
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_new_test_r_approximation_nophase3_1.txt 0.1,0.2,0.5,0.7,1 25 5 60
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_new_test_r_approximation_nophase3_2.txt 0.5 10,20,25,30,40,50,60,70 5 60
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_old_test_r_approximation_nophase3_1.txt 0.1,0.2,0.5,0.7,1 25 5 60
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_old_test_r_approximation_nophase3_2.txt 0.5 10,20,25,30,40,50,60,70 5 60
#python get_JS_runtime_all_methods.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_all_methods_1.txt 0.1,0.2,0.5,0.7,1 25 5 60
#python get_JS_runtime_all_methods.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_all_methods_2.txt 0.5 10,20,25,30,40,50,60,70 5 60


#python get_JS_runtime_all_methods_new_r.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_test_new_r_approximation_1.txt 0.1,0.2,0.3,0.5,0.7,1 20 5 70
#python get_JS_runtime_all_methods_new_r.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_test_new_r_approximation_2.txt 0.2 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_test_new_r_approximation_nophase3_1.txt 0.1,0.2,0.3,0.5,0.7,1 20 5 70
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_test_new_r_approximation_nophase3_2.txt 0.2 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_test_old_r_approximation_nophase3_1.txt 0.1,0.2,0.3,0.5,0.7,1 20 5 70
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_test_old_r_approximation_nophase3_2.txt 0.2 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_all_methods_1.txt 0.1,0.2,0.3,0.5,0.7,1 20 5 70
#python get_JS_runtime_all_methods.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_all_methods_2.txt 0.2 10,20,30,40,50,60,70 5 70


#python get_JS_runtime_all_methods_new_r.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_test_new_r_approximation_1.txt 0.1,0.2,0.5,0.7,1 60 5 70
#python get_JS_runtime_all_methods_new_r.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_test_new_r_approximation_2.txt 1 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_test_new_r_approximation_nophase3_1.txt 0.1,0.2,0.5,0.7,1 60 5 70
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_test_new_r_approximation_nophase3_2.txt 1 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_test_old_r_approximation_nophase3_1.txt 0.1,0.2,0.5,0.7,1 60 5 70
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_test_old_r_approximation_nophase3_2.txt 1 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_all_methods_1.txt 0.1,0.2,0.5,0.7,1 60 5 70
#python get_JS_runtime_all_methods.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_all_methods_2.txt 1 10,20,30,40,50,60,70 5 70


#python get_JS_runtime_all_methods_new_r.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_test_new_r_approximation_1.txt 0.1,0.2,0.5,0.7,1 30 5 70
#python get_JS_runtime_all_methods_new_r.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_test_new_r_approximation_2.txt 1 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_test_new_r_approximation_nophase3_1.txt 0.1,0.2,0.5,0.7,1 30 5 70
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_test_new_r_approximation_nophase3_2.txt 1 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_test_old_r_approximation_nophase3_1.txt 0.1,0.2,0.5,0.7,1 30 5 70
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_test_old_r_approximation_nophase3_2.txt 1 10,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_all_methods_1.txt 0.1,0.2,0.5,0.7,1 30 5 70
#python get_JS_runtime_all_methods.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_all_methods_2.txt 1 10,20,30,40,50,60,70 5 70

#python get_JS_runtime_all_methods_new_r.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_test_new_r_approximation_1.txt 0.02,0.03,0.1,0.2,0.5,0.7,1 20 5 70
#python get_JS_runtime_all_methods_new_r.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_test_new_r_approximation_2.txt 0.02 5,10,15,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_test_new_r_approximation_nophase3_1.txt 0.02,0.03,0.1,0.2,0.5,0.7,1 20 5 70
#python get_JS_runtime_all_methods_new_r_nophase3.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_test_new_r_approximation_nophase3_2.txt 0.02 5,10,15,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_test_old_r_approximation_nophase3_1.txt 0.02,0.03,0.1,0.2,0.5,0.7,1 20 5 70
#python get_JS_runtime_all_methods_old_r_nophase3.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_test_old_r_approximation_nophase3_2.txt 0.02 5,10,15,20,30,40,50,60,70 5 70

#python get_JS_runtime_all_methods_3.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_all_methods_1.txt 1 20 5 70
#python get_JS_runtime_all_methods_2.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_more_results_greedLB.txt 0.7 20 5 70
#python get_JS_runtime_all_methods.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_all_methods_1.txt 0.01,0.02,0.03,0.1,0.2,0.5,0.7,1 20 5 70
#python get_JS_runtime_all_methods.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_all_methods_2.txt 0.02 5,10,15,20,30,40,50,60,70 5 70
#python get_JS_runtime_all_methods.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_all_methods_2.txt 0.02 5,10,15,20,30,40,50,60,70 5 70
# -----------------


#python Lower_JC_greedy.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_s1_JS_runtime_ILP_1.txt 0.05,0.1,0.2,0.5 4

#python Lower_JC_greedy.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_s1_JS_runtime_ILP_1.txt 0.05,0.1,0.2,0.5 4

#python Lower_JC_greedy.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_s1_JS_runtime_ILP_1.txt 0.05,0.1,0.2,0.5 4

#python Lower_JC_greedy.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_s1_JS_runtime_ILP_1.txt 0.05,0.1,0.2,0.5 4

#python Lower_JC_greedy.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_s1_JS_runtime_ILP_1.txt 0.05,0.1,0.2,0.5 4

#python Lower_JC_greedy.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_s1_JS_runtime_ILP_1.txt 0.05,0.1,0.2,0.5 4

#python Lower_JC_greedy.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_s1_JS_runtime_ILP_1.txt 


#______________



#python test_greedyLB_r_approximation.py ./data/dblp/ego_list_1.pkl 2,3,4,5,6,7,8,9,10,11,12,13,14,15

#python test_greedyLB_r_approximation.py ./data/fb-friends/ego_list_2.pkl 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100
#python test_greedyLB_r_approximation.py ./data/fb-friends/ego_list_1.pkl 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100
#python test_greedyLB_r_approximation.py ./data/fb-friends/ego_list_2.pkl 30,35,40,45,50
#python test_greedyLB_r_approximation.py ./data/fb-friends/ego_list_2.pkl 55,60,65,70,75
#python test_greedyLB_r_approximation.py ./data/fb-friends/ego_list_2.pkl 80,85,90,95,100
#python test_greedyLB_r_approximation.py ./data/epinions/ego_list_1.pkl 2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90

#python test_greedyLB_r_approximation.py ./data/wiki/ego_list_1.pkl 2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90


#python test_greedyLB_r_approximation.py ./data/stackoverflow/ego_list_1.pkl 2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90
#python test_greedyLB_r_approximation.py ./data/STACKEXCH/ego_list_1.pkl 2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90
#python test_greedyLB_r_approximation.py ./data/worldcup/ego_list_1.pkl 2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90

#python test_two_heuristics.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_sketch_test_r.txt ./data/epinions/label_1.txt 23 1 5 5,10,20,30,40,50,60 false
#python test_two_heuristics.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_sketch_test_q.txt ./data/epinions/label_1.txt 23 1 1,2,3,4,5,10,15,20 30 false
#python test_two_heuristics.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_sketch_test_p.txt ./data/epinions/label_1.txt 23 1,2,3,4,5,6,7,8 5 30 false

#python test_two_heuristics.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_sketch_test_r.txt ./data/wiki/label_1.txt 11 1 5 5,10,20,30,40,50,60 false
#python test_two_heuristics.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_sketch_test_q.txt ./data/wiki/label_1.txt 11 1 1,2,3,4,5,10,15,20 30 false
#python test_two_heuristics.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_sketch_test_p.txt ./data/wiki/label_1.txt 11 1,2,3,4,5,6,7,8 5 30 false

#python test_two_heuristics.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_sketch_test_r.txt ./data/stackoverflow/label_1.txt 22 1 5 5,10,20,30,40,50,60 false
#python test_two_heuristics.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_sketch_test_q.txt ./data/stackoverflow/label_1.txt 22 1 1,2,3,4,5,10,15,20 30 false
#python test_two_heuristics.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_sketch_test_p.txt ./data/stackoverflow/label_1.txt 22 1 1,2,3,4,5,6,7,8 5 30 false

#python test_two_heuristics.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_sketch_test_r.txt ./data/STACKEXCH/label_1.txt 22 1 5 5,10,20,30,40,50,60 false
#python test_two_heuristics.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_sketch_test_q.txt ./data/STACKEXCH/label_1.txt 22 1 1,2,3,4,5,10,15,20 30 false
#python test_two_heuristics.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_sketch_test_p.txt ./data/STACKEXCH/label_1.txt 22 1 1,2,3,4,5,6,7,8 5 30 false

#python test_two_heuristics.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_sketch_test_r.txt ./data/worldcup/label_1.txt 15 1 5 10,20,30,40,50,60 false
#python test_two_heuristics.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_sketch_test_q.txt ./data/worldcup/label_1.txt 15 1 5,10,15,20 30 false
#python test_two_heuristics.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_sketch_test_p.txt ./data/worldcup/label_1.txt 15 1 4,5,6,7,8 5 30 false


# -------------------------------

#python DP_EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_DP_EXACT_SCUB_1.txt ./data/dblp/label_1.txt 5
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_DP_EXACT_SCUB_2.txt ./data/dblp/label_2.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_DP_EXACT_SCUB_3.txt ./data/dblp/label_3.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_DP_EXACT_SCUB_4.txt ./data/dblp/label_4.txt 6
#python DP_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_DP_EXACT_SCUB_5.txt ./data/dblp/label_5.txt 8


#python TD_EXACT_SCUB.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_TD_EXACT_SCUB_1.txt ./data/dblp/label_1.txt 5
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_TD_EXACT_SCUB_2.txt ./data/dblp/label_2.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_TD_EXACT_SCUB_3.txt ./data/dblp/label_3.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_TD_EXACT_SCUB_4.txt ./data/dblp/label_4.txt 6
#python TD_EXACT_SCUB.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_TD_EXACT_SCUB_5.txt ./data/dblp/label_5.txt 8

#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_1.txt ./data/fb-friends/label_1.txt 14
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_2.txt ./data/fb-friends/label_2.txt 11
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_3.txt ./data/fb-friends/label_3.txt 20
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_4.txt ./data/fb-friends/label_4.txt 19
#python TD_EXACT_SCUB.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_EXACT_SCUB_5.txt ./data/fb-friends/label_5.txt 8

#python top_down_greedy.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_TD_greedy_1.txt ./data/fb-friends/label_1.txt 14
#python top_down_greedy.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_TD_greedy_2.txt ./data/fb-friends/label_2.txt 11
#python top_down_greedy.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_TD_greedy_3.txt ./data/fb-friends/label_3.txt 20
#python top_down_greedy.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_TD_greedy_4.txt ./data/fb-friends/label_4.txt 19
#python top_down_greedy.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_TD_greedy_5.txt ./data/fb-friends/label_5.txt 8

#python top_down_greedy.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_TD_greedy_1.txt ./data/epinions/label_1.txt 23
#python top_down_greedy.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_TD_greedy_2.txt ./data/epinions/label_2.txt 23
#python top_down_greedy.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_TD_greedy_3.txt ./data/epinions/label_3.txt 23
#python top_down_greedy.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_TD_greedy_4.txt ./data/epinions/label_4.txt 23
#python top_down_greedy.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_TD_greedy_5.txt ./data/epinions/label_5.txt 23

#python top_down_greedy.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_TD_greedy_1.txt ./data/wiki/label_1.txt 11
#python top_down_greedy.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_TD_greedy_2.txt ./data/wiki/label_2.txt 10
#python top_down_greedy.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_TD_greedy_3.txt ./data/wiki/label_3.txt 11
#python top_down_greedy.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_TD_greedy_4.txt ./data/wiki/label_4.txt 17
#python top_down_greedy.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_TD_greedy_5.txt ./data/wiki/label_5.txt 25

#python top_down_greedy.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_TD_greedy_1.txt ./data/stackoverflow/label_1.txt 22
#python top_down_greedy.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_TD_greedy_2.txt ./data/stackoverflow/label_2.txt 22
#python top_down_greedy.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_TD_greedy_3.txt ./data/stackoverflow/label_3.txt 22
#python top_down_greedy.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_TD_greedy_4.txt ./data/stackoverflow/label_4.txt 22
#python top_down_greedy.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_TD_greedy_5.txt ./data/stackoverflow/label_5.txt 21

#python top_down_greedy.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_1.txt ./data/STACKEXCH/label_1.txt 22
#python top_down_greedy.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_2.txt ./data/STACKEXCH/label_2.txt 22
#python top_down_greedy.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_3.txt ./data/STACKEXCH/label_3.txt 22
#python top_down_greedy.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_4.txt ./data/STACKEXCH/label_4.txt 16
#python top_down_greedy.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_TD_greedy_5.txt ./data/STACKEXCH/label_5.txt 22

#python top_down_greedy.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_TD_greedy_1.txt ./data/worldcup/label_1.txt 15


#---------------------------

#python test_JS_LBJS.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_1_relative_error_1.txt 0.1 1,2,3,4
#python test_JS_LBJS.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_1_relative_error_2.txt 0.05,0.1,0.15,0.2 3

#python test_JS_LBJS.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_1_relative_error_1.txt 0.1 1,2,3,4
#python test_JS_LBJS.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_1_relative_error_2.txt 0.05,0.1,0.15,0.2 3
#python test_JS_LBJS.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_1_relative_error_3.txt 0.01 1,2,4,6,8,10

#python test_JS_LBJS.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_1_relative_error_1.txt 0.1 1,2,3,4
#python test_JS_LBJS.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_1_relative_error_2.txt 0.05,0.1,0.15,0.2 3

#python test_JS_LBJS.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_1_relative_error_1.txt 0.1 1,2,3,4
#python test_JS_LBJS.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_1_relative_error_2.txt 0.05,0.1,0.15,0.2 3

#python test_JS_LBJS.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_1_relative_error_1.txt 0.1 1,2,3,4
#python test_JS_LBJS.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_1_relative_error_2.txt 0.05,0.1,0.15,0.2 3
#python test_JS_LBJS.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_1_relative_error_3.txt 0.01 1,2,4,6,8,10

#python test_JS_LBJS.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_1_relative_error_1.txt 0.1 1,2,3,4
#python test_JS_LBJS.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_1_relative_error_2.txt 0.05,0.1,0.15,0.2 3

#python test_JS_LBJS.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_1_relative_error_1.txt 0.1 1,2,3,4
#python test_JS_LBJS.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_1_relative_error_2.txt 0.05,0.1,0.15,0.2 3


# ------------------------
#python DP_Greedy.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_greedy_1.txt ./data/dblp/label_1.txt 5
#python DP_Greedy.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_greedy_2.txt ./data/dblp/label_2.txt 6
#python DP_Greedy.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_greedy_3.txt ./data/dblp/label_3.txt 6
#python DP_Greedy.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_greedy_4.txt ./data/dblp/label_4.txt 6
#python DP_Greedy.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_greedy_5.txt ./data/dblp/label_5.txt 8

#python DP_Greedy.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_greedy_1.txt ./data/fb-friends/label_1.txt 14
#python DP_Greedy.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_greedy_2.txt ./data/fb-friends/label_2.txt 11
#python DP_Greedy.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_greedy_3.txt ./data/fb-friends/label_3.txt 20
#python DP_Greedy.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_greedy_4.txt ./data/fb-friends/label_4.txt 19
#python DP_Greedy.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_greedy_5.txt ./data/fb-friends/label_5.txt 8

#python DP_Greedy.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_greedy_1.txt ./data/epinions/label_1.txt 23
#python DP_Greedy.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_greedy_2.txt ./data/epinions/label_2.txt 23
#python DP_Greedy.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_greedy_3.txt ./data/epinions/label_3.txt 23
#python DP_Greedy.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_greedy_4.txt ./data/epinions/label_4.txt 23
#python DP_Greedy.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_greedy_5.txt ./data/epinions/label_5.txt 23

#python DP_Greedy.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_greedy_1.txt ./data/wiki/label_1.txt 11
#python DP_Greedy.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_greedy_2.txt ./data/wiki/label_2.txt 10
#python DP_Greedy.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_greedy_3.txt ./data/wiki/label_3.txt 11
#python DP_Greedy.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_greedy_4.txt ./data/wiki/label_4.txt 17
#python DP_Greedy.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_greedy_5.txt ./data/wiki/label_5.txt 25

#python DP_Greedy.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_greedy_1.txt ./data/stackoverflow/label_1.txt 22
#python DP_Greedy.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_greedy_2.txt ./data/stackoverflow/label_2.txt 22
#python DP_Greedy.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_greedy_3.txt ./data/stackoverflow/label_3.txt 22
#python DP_Greedy.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_greedy_4.txt ./data/stackoverflow/label_4.txt 22
#python DP_Greedy.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_greedy_5.txt ./data/stackoverflow/label_5.txt 21

#python DP_Greedy.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_greedy_1.txt ./data/STACKEXCH/label_1.txt 22
#python DP_Greedy.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_greedy_2.txt ./data/STACKEXCH/label_2.txt 22
#python DP_Greedy.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_greedy_3.txt ./data/STACKEXCH/label_3.txt 22
#python DP_Greedy.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_greedy_4.txt ./data/STACKEXCH/label_4.txt 16
#python DP_Greedy.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_greedy_5.txt ./data/STACKEXCH/label_5.txt 22

#python DP_Greedy.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_greedy_1.txt ./data/worldcup/label_1.txt 15

# ------------------------
#python DP_ILP.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_ILP_1.txt ./data/dblp/label_1.txt 5
#python DP_ILP.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_ILP_2.txt ./data/dblp/label_2.txt 6
#python DP_ILP.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_ILP_3.txt ./data/dblp/label_3.txt 6
#python DP_ILP.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_ILP_4.txt ./data/dblp/label_4.txt 6
#python DP_ILP.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_ILP_5.txt ./data/dblp/label_5.txt 8




# --------------

#python faster_heuristics.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_heuristic_1.txt ./data/dblp/label_1.txt 5
#python faster_heuristics.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_heuristic_2.txt ./data/dblp/label_2.txt 6
#python faster_heuristics.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_heuristic_3.txt ./data/dblp/label_3.txt 6
#python faster_heuristics.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_heuristic_4.txt ./data/dblp/label_4.txt 6
#python faster_heuristics.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_heuristic_5.txt ./data/dblp/label_5.txt 8

#python faster_heuristics.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_heuristic_1.txt ./data/fb-friends/label_1.txt 14
#python faster_heuristics.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_heuristic_2.txt ./data/fb-friends/label_2.txt 11
#python faster_heuristics.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_heuristic_3.txt ./data/fb-friends/label_3.txt 20
#python faster_heuristics.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_heuristic_4.txt ./data/fb-friends/label_4.txt 19
#python faster_heuristics.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_heuristic_5.txt ./data/fb-friends/label_5.txt 8

#python faster_heuristics.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_heuristic_1.txt ./data/epinions/label_1.txt 23
#python faster_heuristics.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_heuristic_2.txt ./data/epinions/label_2.txt 23
#python faster_heuristics.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_heuristic_3.txt ./data/epinions/label_3.txt 23
#python faster_heuristics.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_heuristic_4.txt ./data/epinions/label_4.txt 23
#python faster_heuristics.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_heuristic_5.txt ./data/epinions/label_5.txt 23

#python faster_heuristics.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_heuristic_1.txt ./data/wiki/label_1.txt 11
#python faster_heuristics.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_heuristic_2.txt ./data/wiki/label_2.txt 10
#python faster_heuristics.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_heuristic_3.txt ./data/wiki/label_3.txt 11
#python faster_heuristics.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_heuristic_4.txt ./data/wiki/label_4.txt 17
#python faster_heuristics.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_heuristic_5.txt ./data/wiki/label_5.txt 25

#python faster_heuristics.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_heuristic_1.txt ./data/stackoverflow/label_1.txt 22
#python faster_heuristics.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_heuristic_2.txt ./data/stackoverflow/label_2.txt 22
#python faster_heuristics.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_heuristic_3.txt ./data/stackoverflow/label_3.txt 22
#python faster_heuristics.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_heuristic_4.txt ./data/stackoverflow/label_4.txt 22
#python faster_heuristics.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_heuristic_5.txt ./data/stackoverflow/label_5.txt 21

#python faster_heuristics.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_heuristic_1.txt ./data/STACKEXCH/label_1.txt 22
#python faster_heuristics.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_heuristic_2.txt ./data/STACKEXCH/label_2.txt 22
#python faster_heuristics.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_heuristic_3.txt ./data/STACKEXCH/label_3.txt 22
#python faster_heuristics.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_heuristic_4.txt ./data/STACKEXCH/label_4.txt 16
#python faster_heuristics.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_heuristic_5.txt ./data/STACKEXCH/label_5.txt 22

#python faster_heuristics.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_heuristic.txt ./data/worldcup/label_1.txt 15
#python faster_heuristics.py ./data/TKDD_DATA/ego_list_1.pkl ./data/TKDD_DATA/worldcup_heuristic.txt ./data/TKDD_DATA/label_1.txt 15
####---------

#python test_two_heuristics.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_sketch_test_1.txt ./data/dblp/label_1.txt 5 1 5 40 false
#python test_two_heuristics.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_sketch_test_r.txt ./data/epinions/label_1.txt 23 1 5 5,10,20,30,40,50,60 false


#python test_two_heuristics.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_heuristic_HyperMinHash_1.txt ./data/dblp/label_1.txt 5 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/dblp/ego_list_2.pkl ./data/dblp/dblp_heuristic_HyperMinHash_2.txt ./data/dblp/label_2.txt 6 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/dblp/ego_list_3.pkl ./data/dblp/dblp_heuristic_HyperMinHash_3.txt ./data/dblp/label_3.txt 6 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/dblp/ego_list_4.pkl ./data/dblp/dblp_heuristic_HyperMinHash_4.txt ./data/dblp/label_4.txt 6 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/dblp/ego_list_5.pkl ./data/dblp/dblp_heuristic_HyperMinHash_5.txt ./data/dblp/label_5.txt 8 1 5 20,30,40,50,60 false

#python test_two_heuristics.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_sketch_test_1.txt ./data/fb-friends/label_1.txt 14 1 4 5,10,20,30,40,50,60 false

#python test_two_heuristics.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_heuristic_HyperMinHash_1.txt ./data/fb-friends/label_1.txt 14 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/fb-friends/ego_list_2.pkl ./data/fb-friends/fb_heuristic_HyperMinHash_2.txt ./data/fb-friends/label_2.txt 11 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/fb-friends/ego_list_3.pkl ./data/fb-friends/fb_heuristic_HyperMinHash_3.txt ./data/fb-friends/label_3.txt 20 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/fb-friends/ego_list_4.pkl ./data/fb-friends/fb_heuristic_HyperMinHash_4.txt ./data/fb-friends/label_4.txt 19 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_heuristic_HyperMinHash_5.txt ./data/fb-friends/label_5.txt 8 1 5 20,30,40,50,60 false

#python test_two_heuristics.py ./data/epinions/ego_list_1.pkl ./data/epinions/epinions_heuristic_HyperMinHash_1.txt ./data/epinions/label_1.txt 23 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/epinions/ego_list_2.pkl ./data/epinions/epinions_heuristic_HyperMinHash_2.txt ./data/epinions/label_2.txt 23 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/epinions/ego_list_3.pkl ./data/epinions/epinions_heuristic_HyperMinHash_3.txt ./data/epinions/label_3.txt 23 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/epinions/ego_list_4.pkl ./data/epinions/epinions_heuristic_HyperMinHash_4.txt ./data/epinions/label_4.txt 23 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/epinions/ego_list_5.pkl ./data/epinions/epinions_heuristic_HyperMinHash_5.txt ./data/epinions/label_5.txt 23 1 5 20,30,40,50,60 false

#python test_two_heuristics.py ./data/wiki/ego_list_1.pkl ./data/wiki/wiki_heuristic_HyperMinHash_1.txt ./data/wiki/label_1.txt 11 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/wiki/ego_list_2.pkl ./data/wiki/wiki_heuristic_HyperMinHash_2.txt ./data/wiki/label_2.txt 10 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/wiki/ego_list_3.pkl ./data/wiki/wiki_heuristic_HyperMinHash_3.txt ./data/wiki/label_3.txt 11 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/wiki/ego_list_4.pkl ./data/wiki/wiki_heuristic_HyperMinHash_4.txt ./data/wiki/label_4.txt 17 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/wiki/ego_list_5.pkl ./data/wiki/wiki_heuristic_HyperMinHash_5.txt ./data/wiki/label_5.txt 25 1 5 20,30,40,50,60 false

#python test_two_heuristics.py ./data/stackoverflow/ego_list_1.pkl ./data/stackoverflow/stackoverflow_heuristic_HyperMinHash_1.txt ./data/stackoverflow/label_1.txt 22 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/stackoverflow/ego_list_2.pkl ./data/stackoverflow/stackoverflow_heuristic_HyperMinHash_2.txt ./data/stackoverflow/label_2.txt 22 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/stackoverflow/ego_list_3.pkl ./data/stackoverflow/stackoverflow_heuristic_HyperMinHash_3.txt ./data/stackoverflow/label_3.txt 22 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/stackoverflow/ego_list_4.pkl ./data/stackoverflow/stackoverflow_heuristic_HyperMinHash_4.txt ./data/stackoverflow/label_4.txt 22 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/stackoverflow/ego_list_5.pkl ./data/stackoverflow/stackoverflow_heuristic_HyperMinHash_5.txt ./data/stackoverflow/label_5.txt 21 1 5 20,30,40,50,60 false

#python test_two_heuristics.py ./data/STACKEXCH/ego_list_1.pkl ./data/STACKEXCH/STACKEXCH_heuristic_HyperMinHash_1.txt ./data/STACKEXCH/label_1.txt 22 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/STACKEXCH/ego_list_2.pkl ./data/STACKEXCH/STACKEXCH_heuristic_HyperMinHash_2.txt ./data/STACKEXCH/label_2.txt 22 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/STACKEXCH/ego_list_3.pkl ./data/STACKEXCH/STACKEXCH_heuristic_HyperMinHash_3.txt ./data/STACKEXCH/label_3.txt 22 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/STACKEXCH/ego_list_4.pkl ./data/STACKEXCH/STACKEXCH_heuristic_HyperMinHash_4.txt ./data/STACKEXCH/label_4.txt 16 1 5 20,30,40,50,60 false
#python test_two_heuristics.py ./data/STACKEXCH/ego_list_5.pkl ./data/STACKEXCH/STACKEXCH_heuristic_HyperMinHash_5.txt ./data/STACKEXCH/label_5.txt 22 1 5 20,30,40,50,60 false

#python faster_heuristics_with_HyperMinHash.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/worldcup_heuristic_HyperMinHash.txt ./data/worldcup/label_1.txt 15 1 5 20,30,40,50,60 false

#python 2_approximation.py ./data/dblp/ego_list_1.pkl ./data/dblp/dblp_2_approximation_1.txt ./data/dblp/label_1.txt 5
#python 2_approximation.py ./data/fb-friends/ego_list_1.pkl ./data/fb-friends/fb_2_approximation_1.txt ./data/fb-friends/label_1.txt 14
#python 2_approximation.py ./data/fb-friends/ego_list_5.pkl ./data/fb-friends/fb_2_approximation_5.txt ./data/fb-friends/label_5.txt 8



#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_new_r_appro_TD_scalability_10.txt ./data/worldcup/label_1.txt 2 5 10 1
#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_new_r_appro_TD_scalability_30.txt ./data/worldcup/label_1.txt 3 5 30 1
#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_new_r_appro_TD_scalability_50.txt ./data/worldcup/label_1.txt 3 5 50 1
#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_new_r_appro_TD_scalability_70.txt ./data/worldcup/label_1.txt 4 5 70 1
#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_new_r_appro_TD_scalability_90.txt ./data/worldcup/label_1.txt 3 5 90 1

#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_old_r_appro_TD_scalability_10.txt ./data/worldcup/label_1.txt 2 5 10 1
#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_old_r_appro_TD_scalability_30.txt ./data/worldcup/label_1.txt 3 5 30 1
#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_old_r_appro_TD_scalability_50.txt ./data/worldcup/label_1.txt 3 5 50 1
#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_old_r_appro_TD_scalability_70.txt ./data/worldcup/label_1.txt 4 5 70 1
#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_old_r_appro_TD_scalability_90.txt ./data/worldcup/label_1.txt 3 5 90 1

#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_S_TD_scalability_10.txt ./data/worldcup/label_1.txt 2 1 5 30 false 10 1
#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_S_TD_scalability_30.txt ./data/worldcup/label_1.txt 3 1 5 30 false 30 1
#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_S_TD_scalability_50.txt ./data/worldcup/label_1.txt 3 1 5 30 false 50 1
#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_S_TD_scalability_70.txt ./data/worldcup/label_1.txt 4 1 5 30 false 70 1
#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_S_TD_scalability_90.txt ./data/worldcup/label_1.txt 3 1 5 30 false 90 1

#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_TD_scalability_10.txt ./data/worldcup/label_1.txt 2 10 1
#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_TD_scalability_30.txt ./data/worldcup/label_1.txt 3 30 1
#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_TD_scalability_50.txt ./data/worldcup/label_1.txt 3 50 1
#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_TD_scalability_70.txt ./data/worldcup/label_1.txt 4 70 1
#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/mfixed_H_TD_scalability_90.txt ./data/worldcup/label_1.txt 3 90 1
# ===============================
#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_new_r_appro_TD_scalability_1.txt ./data/worldcup/label_1.txt 4 5 1 2
#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_new_r_appro_TD_scalability_2.txt ./data/worldcup/label_1.txt 3 5 2 2
#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_new_r_appro_TD_scalability_3.txt ./data/worldcup/label_1.txt 4 5 3 2
#python r_approximation_TD_new_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_new_r_appro_TD_scalability_4.txt ./data/worldcup/label_1.txt 3 5 4 2

#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_old_r_appro_TD_scalability_1.txt ./data/worldcup/label_1.txt 4 5 1 2
#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_old_r_appro_TD_scalability_2.txt ./data/worldcup/label_1.txt 3 5 2 2
#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_old_r_appro_TD_scalability_3.txt ./data/worldcup/label_1.txt 4 5 3 2
#python r_approximation_old_TD_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_old_r_appro_TD_scalability_4.txt ./data/worldcup/label_1.txt 3 5 4 2

#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_H_S_TD_scalability_10.txt ./data/worldcup/label_1.txt 4 1 5 30 false 1 2
#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_H_S_TD_scalability_30.txt ./data/worldcup/label_1.txt 3 1 5 30 false 2 2
#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_H_S_TD_scalability_50.txt ./data/worldcup/label_1.txt 4 1 5 30 false 3 2
#python faster_heuristics_with_HyperMinHash_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_H_S_TD_scalability_70.txt ./data/worldcup/label_1.txt 3 1 5 30 false 4 2

#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_H_TD_scalability_10.txt ./data/worldcup/label_1.txt 4 1 2
#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_H_TD_scalability_30.txt ./data/worldcup/label_1.txt 3 2 2
#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_H_TD_scalability_50.txt ./data/worldcup/label_1.txt 4 3 2
#python faster_heuristics_scalability.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/nfixed_H_TD_scalability_70.txt ./data/worldcup/label_1.txt 3 4 2

#===============================
#python faster_heuristics_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_TD_200.txt ./data/worldcup/label_1.txt 5 200
#python faster_heuristics_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_TD_400.txt ./data/worldcup/label_1.txt 5 400
#python faster_heuristics_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_TD_600.txt ./data/worldcup/label_1.txt 5 600
#python faster_heuristics_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_TD_720.txt ./data/worldcup/label_1.txt 5 720

#python faster_heuristics_with_HyperMinHash_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_S_TD_200.txt ./data/worldcup/label_1.txt 5 1 5 30 false 200
#python faster_heuristics_with_HyperMinHash_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_S_TD_400.txt ./data/worldcup/label_1.txt 5 1 5 30 false 400
#python faster_heuristics_with_HyperMinHash_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_S_TD_600.txt ./data/worldcup/label_1.txt 5 1 5 30 false 600
#python faster_heuristics_with_HyperMinHash_test.py ./data/worldcup/ego_list_1.pkl ./data/worldcup/Sketch_test_H_S_TD_720.txt ./data/worldcup/label_1.txt 5 1 5 30 false 720


#python faster_heuristics_sum_embedding.py ./data/dblp/ego_list_1.pkl 5

#\usr\bin\time -v python faster_heuristics_with_HyperMinHash.py ./data/dblp/ego_list_1.pkl ./data/dblp/test.txt ./data/dblp/label_1.txt 5 1 5 30 false >out
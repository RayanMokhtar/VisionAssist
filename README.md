liens utiles : 

https://plmlatex.math.cnrs.fr/6657316991jrgdvcxtmtxy : latex rapport

https://www.canva.com/design/DAG7Y4vhYag/kMWcCppjId9w26r_Cp3d2w/edit : lien canvas architecture globale






documentation : 

https://medium.com/@pinaki.brahma/improve-llm-based-response-through-parent-child-retrieval-strategy-part-2-04ef7982b9a3 



python3 -m piper.download_voices fr_FR-siwis-medium INSTALLER PIPER 
## docs tts : 


TTS piper : https://arxiv.org/html/2512.08006v1 // https://github.com/OHF-Voice/piper1-gpl?tab=readme-ov-file

#Documentation choix TTS : https://www.datacamp.com/blog/best-open-source-text-to-speech-tts-engines?utm_cid=23552157100&utm_aid=188237542530&utm_campaign=230119_1-ps-other~dsa-tofu~ai_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9056593-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~emea-en~dsa~tofu~blog~artificial-intelligence&gad_source=1&gad_campaignid=23552157100&gbraid=0AAAAADQ9WsFrYlpVH5Kp-ES7XFIPO7iEZ&gclid=CjwKCAjwpcTNBhA5EiwAdO1S9tnpoHkvvrWhSv92EZYExFu1k8D_lvjfX5-Hk7QYHOa8dY13zlenBBoCdd0QAvD_BwE

fichier entrainement piper : https://raw.githubusercontent.com/rhasspy/piper/master/TRAINING.md

https://arxiv.org/pdf/2106.06103


lien vers la démo : https://rhasspy.github.io/piper-samples/demo.html


python3 -m piper.download_voices fr_FR-siwis-medium INSTALLER PIPER 


entrainement : finetuning modele : https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md

config jetsonGPT : https://github.com/shahizat/jetsonGPT

### Remarque sur ça 
avant de lancer il faut les modèles et créer les dossiers éventuellement : 
installer_piper dans utils ...  mais ensuite déplacer le modèle dans le bon répertoire





### broker  : 

mqtt explorer à installer pour visualiser messages dans le topic

installer le broker mosquitto via l'installer https://mosquitto.org/download/ : puis lancer mosquitto -v , et en admin sc query mosquitto pour voir état et sc stop mosquitto pour le stopper si jamais le port est pris ou qu'on se retrouve dans des situtations reloues



 ## worker vision ( à automatiser dans un setup.sh par la suiteg)
 1 - wsl installé , ouvrir terminal 
 2 - créer build 
 3 - cd worker_vision/build 
 4 - cmake ..
 5 - make puis ./main





 ## refonte sdépendances 

 pip install -e .[jetson] ou serveur

 uv sync --extra jetson 


 nvidia-smi driver ... => résoudre probleme de dépendance version torch pas compatible avec le driver .. 


 mosquitto_pub -h 127.0.0.1 -p 1883 -t "results/stt" -m '{"session_id":"123","resultat_stt":{"texte":"bonjour"}}' -q 1
 mosquitto_sub -h 127.0.0.1 -p 1883 -t "#" -v

sudo apt install -y nvidia-driver-550# maj driver 

// vllm doc suggère : uv
 pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 // à tester si ça marche pas

 commande lancement vllm : vllm serve ./langage/modeles/Qwen3.5-27B-4bit-bitsandbytes --max-model-len 8192 --gpu-memory-utilization 0.95 --enforce-eager

 ajouter allow_anonymous dans broker
 listener 1883 0.0.0.0 dans etc config
 ### Carte Gemalto  : 
 sudo apt install pcscd pcsc-tools swig python3-dev libpcsclite-dev
 pip install pyscard (Dans un venv)

Vérifier que le lecteur est détecté :
pcsc_scan

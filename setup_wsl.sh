
#Add to ~/.bashrc
#configure x11 display
export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0

#alias ipythonqt="jupyter qtconsole --ConsoleWidget.font_size=6 --style=material"
alias ipythonqt="jupyter qtconsole --ConsoleWidget.font_size=6 --style=material --IPythonWidget.gui_completion=droplist"
alias ipymonoqt="jupyter qtconsole --ConsoleWidget.font_size=6 --style=monokai --IPythonWidget.gui_completion=droplist"
alias ipyzenqt="jupyter qtconsole --ConsoleWidget.font_size=6 --style=zenburn --IPythonWidget.gui_completion=droplist"
alias ipyliteqt="jupyter qtconsole --ConsoleWidget.font_size=6 --style=material_light --IPythonWidget.gui_completion=droplist"
alias ipyoceanqt="jupyter qtconsole --ConsoleWidget.font_size=6 --style=base16_ocean_dark --IPythonWidget.gui_completion=droplist"

#these may fail without proper admin config in visudo
sudo /usr/sbin/service cron start

#Add G Drive
#Mount google drive
if mount | grep /mnt/g > /dev/null; then
  echo "G Drive Mounted!"
else
  echo "Mounting Share Drive!"
  sudo mount -t drvfs G: /mnt/g
fi

#Links:
~/dropbox -> '/mnt/c/Users/Sup/Ottermatics Dropbox'
~/nept_proj -> /home/olly/dropbox/Projects/Neptunya/
~/ottermatics -> /home/olly/dropbox/Ottermatics
~/ottermaticslib -> /home/olly/dropbox/Ottermatics/ottermaticslib/
~/smx_proj -> /home/olly/dropbox/Projects/SMART_X/
~/.aws -> /mnt/c/Users/Sup/.aws
~/.azure -> /mnt/c/Users/Sup/.azure


#Add to /etc/wsl.conf
[boot]
systemd=true

[automount]
options="metadata"




#add to crontab
#Cron `crontab -e`
*/30 * * * * /usr/bin/run-one /bin/bash /home/olly/syncthing.sh

#synctab ~/sync_thing.sh
echo "Syncing at `date`" >> /home/olly/nept_proj/.gdrive_sync.log
rsync -u --no-t -O --exclude '.*' --exclude '*.pyc' --exclude '__*__' --exclude 'ray_results*' --exclude 'projects_legal' --exclude 'invoices' --min-size=102>
echo "Done Syncing at `date`" >> /home/olly/nept_proj/.gdrive_sync.log
echo "Done Syncing at `date`" >> /mnt/g/Shared\ drives/neptunya@ottermatics/gdrive_sync.log

#install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash

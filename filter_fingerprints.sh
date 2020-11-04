#!/bin/bash
# Author: Ian Stroszeck
# Updated: Bad3r@pm.me
# Extract all PNGs from NIST doc
# PRECONDITION: NIST Torrent should be extracted as is into a folder called:
# NIST_Fingerprints
: '
ls should have the following set up prior to script execution:
fileMover.sh (this script)
NIST_Fingerprints (zip)
'
# Define File paths
dir=$(pwd) 
FINGERPRINT=$dir/Fingerprint/
NIST=$dir/Fingerprint/NIST_Fingerprints/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/*
TRAIN_S=$dir/Fingerprint/TRAIN/S/
TRAIN_F=$dir/Fingerprint/TRAIN/F/mju

# Make folders
mk_folders() {
	cd $FINGERPRINT
	mkdir NIST_Fingerprints
	mkdir TEST
	mkdir TRAIN
	cd $FINGERPRINT/TRAIN
	mkdir S
	mkdir F
}
unzip_NIST() {
	cd $FINGERPRINT
	unzip -d ./NIST_Fingerprints $dir/NISTSpecialDatabase4GrayScaleImagesofFIGS.zip
}
# Get PNG files
get_pngs() {
# Find every "s" png file
s=$(ls | grep -E 's[0-9]{4}_[0-9]{2}.png')
# Find every "f" png file
f=$(ls | grep -E 'f[0-9]{4}_[0-9]{2}.png')
}
# Move PNG files (s & f) to TEST folder
mv_pngs() {
for s_var in $s;do
	old_s="$folder$s_var";
	train_s="$TRAIN_S$s_var";
	#echo "mv $old_s $train_s" # Testing
	mv $old_s $train_s;
	#read # Pause for testing
done
for f_var in $f;do
	old_f="$folder$f_var";
	train_f="$TRAIN_F$f_var";
	#echo "mv $old_f $train_f" # Testing
	mv $old_f $train_f;
	#read # Pause for testing
done
}
main() {
	# Make S & F folders
	mk_folders
	unzip_NIST
	for folder in $NIST;do
	 folder="$folder/"
	 cd $folder;
	 echo '*****************************'; # Separate
	 echo "Now in $folder"; # Testing
	 get_pngs;
	 #echo "All of S: $s"; # Testing
	 echo 'All s & f png files collected';
	 mv_pngs;
	 echo 'All png files moved to TRAIN folder';
	 echo '*****************************'; # Separate
	done
}
main
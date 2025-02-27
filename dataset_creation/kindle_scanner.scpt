(*
AppleScript Scanner for Kindle App

Instructions: 
1. Set dir where to save files, page scans will show up as "bookName - Page #.jpg" in subdir "saveToLocation + bookName"
2. Open the kindle app on your computer, and open book to the page you want to start scanning on
3. Set page to one-column format, large line-spacing, adjust font size
4. Window size should adjust automatically, but in the case that you need to adjust it, see below and use shift-command-control-4 to find size
5. Set pageCount to number of pages to scan (page flips are automated)
6. Run script by pressing play button
*)


-- # set dir where jpg scans will be saved
set saveToLocation to "/Users/xx/Desktop/book_scans/"
-- # book name + translator
set bookName to "TheBrothersKaramazov_Katz"
-- # num of pages to iterate + scan
set pageCount to 300

set pathFileName to saveToLocation & bookName
tell application "Kindle" to activate

delay 0.5

tell application "System Events" to ¬
	tell window 1 of application process "Kindle" to ¬

		-- # reposition window to this size/loc on screen
		set {position, size} to {{250, 25}, {919, 870}}
repeat with i from 1 to pageCount
	
	set shellCMD to ¬
		-- # screenshot window of these params
		"screencapture -R297,81,855,785 -t jpg '" & ¬

		-- # save screenshot
		pathFileName & " - Page " & ¬
		i & " of " & pageCount & ".jpg'"
	
	do shell script shellCMD
	delay 1.5
	--  # Press right-arrow key.
	tell application "System Events" to key code 124
	delay 1.5
	
end repeat

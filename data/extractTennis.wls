#!/usr/bin/env wolframscript
(* ::Package:: *)

(*Import tournament dataset*)
tourneys=Import[NotebookDirectory[]<>"tournaments_1877-2017_unindexed.csv"];
tourneys//Dimensions
tourneyHeader=Thread[tourneys[[1]]->Range[Length@tourneys[[1]]]]


(*Create lookup table id->day*)
tourneyDays=Quiet@Map[ToString[#[[1]]]<>"-"<>ToString[#[[2]]]->(AbsoluteTime[#[[{1,3,4}]]] - AbsoluteTime["1800-01-01"])/86400&,
	tourneys[[2;;-1,{"tourney_year","tourney_id","tourney_month","tourney_day"}/.tourneyHeader]]];


(*Import match dataset*)
table = Import[StringJoin[NotebookDirectory[], "match_scores_1991-2016_unindexed.csv"]]; 
Dimensions[table]
header = Thread[table[[1]] -> Range[Length[table[[1]]]]]


(*Pick recent games*)
isRecent[str_]:=Return[str=="2014"||str=="2015"||str=="2016"]
recent=Select[table,isRecent[StringTake[#[[1]],4]]&];
recent=table; (*just pick all*)
recent//Dimensions


(*Select important columns*)
cols={"tourney_year_id","winner_name","loser_name","winner_games_won","loser_games_won"};
scores=recent[[All,cols/.header]];
scores//Dimensions





(*Pick the players with the most games. Pick the games with at least one of such a player.*)
activePlayers=Reverse[SortBy[Tally@Join[recent[[All,"winner_name"/.header]],recent[[All,"loser_name"/.header]]],Last]][[1;;10]]
importantPlayers=activePlayers[[All,1]];
importantGames=Select[scores,ContainsAny[importantPlayers,#[[Position[cols,#][[1,1]]&/@{"winner_name","loser_name"}]]]&];

(*Replace tourney id by day*)
importantGames[[All,1]]=importantGames[[All,1]]/.tourneyDays;
importantGames=SortBy[importantGames,First];
importantGames//Dimensions


(*Export to csv*)
Export[NotebookDirectory[]<>"tennis2.csv",Prepend[importantGames,cols/."tourney_year_id"->"day"]]

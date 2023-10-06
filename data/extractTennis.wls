#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*(*Import tournament dataset*)*)
(*tourneys=Import[NotebookDirectory[]<>"tournaments_1877-2017_unindexed.csv"];*)
(*tourneys//Dimensions*)
(*tourneyHeader=Thread[tourneys[[1]]->Range[Length@tourneys[[1]]]]*)


tourneyDays=Map[ToString[#[[1]]]<>"-"<>ToString[#[[2]]]->(AbsoluteTime[#[[{1,3,4}]]] - AbsoluteTime["1800-01-01"])/86400&,
	tourneys[[2;;-1,{"tourney_year","tourney_id","tourney_month","tourney_day"}/.tourneyHeader]]];


(* ::Input:: *)
(*(*Import dataset*)*)
(*table=Import[NotebookDirectory[]<>"match_scores_1991-2016_unindexed.csv"];*)
(*table//Dimensions*)
(*header=Thread[table[[1]]->Range[Length@table[[1]]]]*)


(* ::Input:: *)
(*(*Pick recent games*)*)
(*isRecent[str_]:=Return[str=="2014"||str=="2015"||str=="2016"]*)
(*recent=Select[table,isRecent[StringTake[#[[1]],4]]&];*)
(*recent//Dimensions*)


recent=table;


(* ::Input:: *)
(*(*Select important columns*)*)
(*cols={"tourney_year_id","winner_name","loser_name","winner_games_won","loser_games_won"};*)
(*scores=recent[[All,cols/.header]];*)
(*scores//Dimensions*)


(* ::Input:: *)
(*(*Pick the players with the most games. Pick the games with at least one of such a player.*)*)
(*activePlayers=Reverse[SortBy[Tally@Join[recent[[All,"winner_name"/.header]],recent[[All,"loser_name"/.header]]],Last]][[1;;10]]*)
(*importantPlayers=activePlayers[[All,1]];*)
(*importantGames=Select[scores,ContainsAny[importantPlayers,#[[1;;2]]]&];*)
(*importantGames[[All,1]]=importantGames[[All,1]]/.tourneyDays;*)
(**)
(*importantGames//Dimensions*)


(* ::Input:: *)
(*(*Export to csv*)*)
(*Export[NotebookDirectory[]<>"tennis2.csv",Prepend[importantGames,cols]]*)


(* ::Text:: *)
(* *)

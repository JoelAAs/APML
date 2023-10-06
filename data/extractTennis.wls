#!/usr/bin/env wolframscript
(* ::Package:: *)

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


(* ::Input:: *)
(*(*Select important columns*)*)
(*cols={"winner_name","loser_name","winner_games_won","loser_games_won"};*)
(*scores=recent[[All,cols/.header]];*)
(*scores//Dimensions*)


(* ::Input:: *)
(*(*Pick the players with the most games. Pick the games with at least one of such a player.*)*)
(*activePlayers=Reverse[SortBy[Tally@Join[recent[[All,8]],recent[[All,11]]],Last]][[1;;30]]*)
(*importantPlayers=activePlayers[[All,1]];*)
(*importantGames=Select[scores,ContainsAny[importantPlayers,#[[1;;2]]]&];*)
(*importantGames//Dimensions*)


(* ::Input:: *)
(*(*Export to csv*)*)
(*Export[NotebookDirectory[]<>"tennis.csv",Prepend[importantGames,cols]]*)

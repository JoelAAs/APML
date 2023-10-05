#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*table=Import[NotebookDirectory[]<>"match_scores_1991-2016_unindexed.csv"];*)
(*table//Dimensions*)


(* ::Input:: *)
(*header=Thread[table[[1]]->Range[Length@table[[1]]]]*)


(* ::Input:: *)
(*(*Pick recent games*)*)
(*isRecent[str_]:=Return[str=="2014"||str=="2015"||str=="2016"]*)
(*recent=Select[table,isRecent[StringTake[#[[1]],4]]&];*)
(*recent//Dimensions*)


(* ::Input:: *)
(*scores=recent[[All,{"winner_name","loser_name","winner_games_won","loser_games_won"}/.header]];*)
(*scores//Dimensions*)


(* ::Input:: *)
(*activePlayers=Reverse[SortBy[Tally@Join[recent[[All,8]],recent[[All,11]]],Last]][[1;;30]]*)
(*importantPlayers=activePlayers[[All,1]];*)


(* ::Input:: *)
(*importantGames=Select[scores,ContainsAny[importantPlayers,#[[1;;2]]]&];*)
(*importantGames//Dimensions*)


(* ::Input:: *)
(*Export[NotebookDirectory[]<>"tennis.csv",importantGames]*)

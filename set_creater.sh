
cluster=('mobley_1017962' 'mobley_1075836' 'mobley_1502181' 'mobley_1592519' 'mobley_1662128' 'mobley_1717215' 'mobley_1722522' 'mobley_1735893' 'mobley_1827204' 'mobley_1875719' 'mobley_1881249' 'mobley_1899443' 'mobley_1963873' 'mobley_1967551' 'mobley_1976156' 'mobley_2143011' 'mobley_2146331' 'mobley_2294995' 'mobley_2341732' 'mobley_2364370' 'mobley_2390199' 'mobley_2487143' 'mobley_2493732' 'mobley_2607611' 'mobley_2929847' 'mobley_299266' 'mobley_303222' 'mobley_3034976' 'mobley_3572203' 'mobley_3573480' 'mobley_3690931' 'mobley_3709920' 'mobley_3867265' 'mobley_3982371' 'mobley_4043951' 'mobley_4465023' 'mobley_4694328' 'mobley_4762983' 'mobley_4792268' 'mobley_4924862' 'mobley_5310099' 'mobley_5346580' 'mobley_5390332' 'mobley_5471704' 'mobley_5627459' 'mobley_5890803' 'mobley_5952846' 'mobley_6091882' 'mobley_6474572' 'mobley_6619554' 'mobley_6632459' 'mobley_6854178' 'mobley_6929123' 'mobley_6973347' 'mobley_6981465' 'mobley_7203421' 'mobley_7375018' 'mobley_7417968' 'mobley_7455579' 'mobley_7573149' 'mobley_766666' 'mobley_7758918' 'mobley_7859387' 'mobley_7943327' 'mobley_8011706' 'mobley_8048190' 'mobley_820789' 'mobley_8337977' 'mobley_8427539' 'mobley_8467917' 'mobley_8492526' 'mobley_8514745' 'mobley_8573194' 'mobley_8578590' 'mobley_859464' 'mobley_8739734' 'mobley_8754702' 'mobley_9112978' 'mobley_9197172' 'mobley_9407874' 'mobley_9617923' 'mobley_9913368')

rm cluster_0.sdf
for i in "${cluster[@]}"
do
    echo $i
    cat input/FreeSolv/$i.sdf >> cluster_0.sdf
done
```
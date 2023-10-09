while IFS= read -r line; do
    echo "Kill Process ${line}"
    kill ${line}
done < "./log/${1}/.save_pid"
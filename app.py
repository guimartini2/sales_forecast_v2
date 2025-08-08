```python
# Replenishment logic based on manual init_inv
on_hand_list = []
replenishment = []
prev_on = init_inv
for _, row in df_fc.iterrows():
    D = row[forecast_label]
    # Inventory target = demand * WOC
    target_inv = D * woc_target
    Q = max(target_inv - prev_on, 0)
    on_hand_list.append(int(prev_on))
    replenishment.append(int(Q))
    # Update on-hand inventory after meeting demand
    prev_on = prev_on + Q - D

# Assign computed fields to DataFrame
df_fc['On_Hand_Begin'] = on_hand_list
df_fc['Replenishment'] = replenishment```

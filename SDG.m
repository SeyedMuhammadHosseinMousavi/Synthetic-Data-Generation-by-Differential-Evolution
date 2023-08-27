function agent_cost = SDG(agent, x_data, y_data)
agent_cost = sum((polyval(agent, x_data) - y_data).^2);
end 
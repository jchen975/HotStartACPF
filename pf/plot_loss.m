function plot_loss(str, T) % assuming va, vm trained separately
    types = ["va", "vm"];
    for t = types
        t = char(t);
        dir = 'D:\Work\Research\Cluster Results\';
        load([dir, str, '\', str, '_', t, '_loss_T=', num2str(T), '.mat']);
        numBatch = int64(length(trainLoss) / length(valLoss));
        temp_valLoss = zeros(size(trainLoss));
        for i = 1:length(valLoss)  % epochs
            for j = 1:numBatch
                temp_valLoss(j+(i-1)*numBatch) = valLoss(i);
            end
        end
        itr = 1:length(trainLoss);
        
        h = figure;
        semilogy(itr, trainLoss);
        hold on;
        semilogy(itr, temp_valLoss);
        xlabel('Iterations');
        ylabel('Log Loss');
        legend('Training Loss', 'Validation Loss');
        grid on
        set(gca, 'FontName', 'Times New Roman');
        hold off;
        
        set(h,'Units','Inches');
        pos = get(h,'Position');
        set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
        print(h, [dir, str, '_', t, '_loss_T=', num2str(T),'.pdf'],'-dpdf','-r0')
    end
end
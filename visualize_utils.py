import numpy as np
import matplotlib.pyplot as plt

def visualize_failed_mols_w_hydrogens(info, title=None):
    # get all batches
    atm2color = {'C':'black', 'O':'blue', 'N':'green', 'F':'yellow', 'H':'orange'}
    bnd2color = {1:'yellow', 2:'red', 3:'brown', 4:'orange', 1.5:'orange'}
    for failed_batch_info in info:
        if len(failed_batch_info) != 5: continue # some batches might be broken
        field, candidaten_bnds,actual_bnd,atm_symb,atm_pos = failed_batch_info

        # create a 3d plot from the Nx3 array atm_bnd_pos
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # for atm in atm_pos:
        colors = [atm2color[a] for a in atm_symb]
        ax.scatter(atm_pos[:,0], atm_pos[:,1], atm_pos[:,2], c=colors, marker='o',s=300)
        for bnd in actual_bnd:
            bnd_inds = [bnd[0], bnd[1]]
            line = atm_pos[bnd_inds]
            dist = np.linalg.norm(line[0]-line[1])
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=bnd2color[bnd[2]], linewidth=3)
            bond_position = (line[0]+line[1])/2
            # ax.text(bond_position[0], bond_position[1], bond_position[2], "{:.3f}".format(dist),  fontsize=20)
        # for bnd in candidaten_bnds:
        if title is not None: plt.title(title)
        plt.show()


def visualize_mol(plot_bnd=0, threshold=0.7, plot_all_atms=False, plot_all_bnd=False, field=None, atm_pos=None, atm_symb=None,actual_bnd=None,
                  atms_required=None, bins=None, batch_ind=None, x_grid=None,y_grid=None,z_grid=None, annotate_atm_no=False, timestep=None, restrict_mol=False,
                  avg_z=True, data='QM9', subplot_inds=None, fig=None,scatter_size=300, title=None, remove_colorbar=False, density_scatter_size=20,y_label="", save_fig_name="", set_legend=False,
                  show_reference=True, automatic_thresh=False, reference='both',no_fld_vals=False,
                  edgecolor='white', ax=None, linewidth=0.5):
        # create a 3d plot from the Nx3 array atm_bnd_pos
    if ax is not None:
        show_plot=False
    elif fig is not None:
        ax = fig.add_subplot(*subplot_inds, projection='3d')
        show_plot=False
    else:
        show_plot=True
        fig = plt.figure(dpi=350)
        fig.set_size_inches(11.5, 10.5)
        ax = fig.add_subplot(111, projection='3d')
    # for atm in atm_pos:
    if data == 'GEOM':
        if plot_all_bnd: inds_ = [5,6,7]
        elif plot_all_atms: inds_ = [0,1,2,3,4]
        else: inds_ = plot_bnd
    elif data == 'QM9':
        if plot_all_bnd: inds_ = [8,9,10]
        elif plot_all_atms: inds_ = [0,1,2,3,4,5,6,7]
        else: inds_ = plot_bnd
    if automatic_thresh: threshold = binary_search(field[inds_].flatten(), 200)
    if no_fld_vals: threshold = 1.1
    min_, max_ = np.min(atm_pos, axis=0) - 0.8, np.max(atm_pos, axis=0) + 0.8


    if "GEOM" in data:
        atm2color = {'C':'grey', 'N':'blue', 'O':'red', 'F':'green', 'P':'purple', 'S':'olive', 'Cl':'mediumseagreen', 'H':'white', 'X':'pink', 'Y':'darkviolet'}
        bnd2color = {1:'orange', 2:'red', 3:'brown', 4:'orange'}
    else:
        atm2color = {'C':'grey', 'O':'red', 'N':'blue', 'F':'green', 'H':'white', 'X':'pink', 'Y':'darkviolet'}
        bnd2color = {1:'orange', 2:'red', 3:'brown', 4:'purple'}        
    ax.set_facecolor('black')
    ax.axis('off')
    transp_fld = 0.6 if show_reference else 1
    if show_reference:
        colors = [atm2color[a] for a in atm_symb]
        if reference == 'both' or reference == 'atoms':
            if set_legend:
                for atm in set(atm_symb):
                    atm_pos_ = atm_pos[np.array(atm_symb) == atm]
                    colors_ = [atm2color[atm]] * len(atm_pos_)
                    ax.scatter(atm_pos_[:,0], atm_pos_[:,1], atm_pos_[:,2], c=colors_, 
                               marker='o',s=scatter_size,  alpha=0.5,linewidth=0.2,
                                label=atm, edgecolor=edgecolor)
            else:
                ax.scatter(atm_pos[:,0], atm_pos[:,1], atm_pos[:,2], c=colors, marker='o',s=scatter_size,  alpha=0.5,linewidth=0.2, edgecolor=edgecolor)
            if annotate_atm_no:
                for ind, atm_p in enumerate(atm_pos):
                    ax.text(atm_p[0], atm_p[1], atm_p[2], "{}".format(ind),  fontsize=15)
        set_bond_labels = set([])
        if reference == 'both' or reference == 'bonds':
            for bnd in actual_bnd:
                bnd_inds = [bnd[0], bnd[1]]
                if bnd[2] not in set_bond_labels and set_legend:
                    set_bond_labels.add(bnd[2])
                    line = atm_pos[bnd_inds]
                    dist = np.linalg.norm(line[0]-line[1])
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], color=bnd2color[bnd[2]], linewidth=linewidth, label=f"Bond {bnd[2]}")
                    bond_position = (line[0]+line[1])/2
                else:    
                    line = atm_pos[bnd_inds]
                    dist = np.linalg.norm(line[0]-line[1])
                    ax.plot(line[:, 0], line[:, 1], line[:, 2], color=bnd2color[bnd[2]], linewidth=linewidth)
                    bond_position = (line[0]+line[1])/2
        # ax.text(bond_position[0], bond_position[1], bond_position[2], "{:.3f}".format(dist),  fontsize=20)
    scatter = None
    if plot_all_atms:
        for plot_bnd in range(len(atm2color) -2 if 'X' in atm2color else len(atm2color)):
            if restrict_mol:
                plot_inds =  (x_grid > min_[0]) * (x_grid < max_[0]) * (y_grid > min_[1]) * (y_grid < max_[1]) * (z_grid > min_[2]) * (z_grid < max_[2])
                if threshold is None: threshold = 0.7 * np.max(field[plot_bnd][plot_inds])
                plot_inds = plot_inds * (field[plot_bnd]>threshold)

                # (field[plot_bnd]>threshold) *
                x = x_grid[plot_inds]
                y = y_grid[plot_inds]
                z = z_grid[plot_inds]
                scatter =ax.scatter(x,y,z,c=field[plot_bnd][plot_inds],
                                     edgecolor='black',s=density_scatter_size,
                                     alpha=transp_fld, cmap='viridis')
            else:
                x = x_grid[field[plot_bnd]>threshold]
                y = y_grid[field[plot_bnd]>threshold]
                z = z_grid[field[plot_bnd]>threshold]
                alpha=field[plot_bnd][field[plot_bnd]>threshold]**2 if len(field[plot_bnd][field[plot_bnd]>threshold]) else 1
                size = alpha * 50
                scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold],  
                                    s=density_scatter_size,alpha=transp_fld, cmap='viridis')
        if not remove_colorbar: plt.colorbar(scatter)
    elif plot_all_bnd:
        for plot_bnd in range(len(atm2color) - 2, field.shape[0]):
            x = x_grid[field[plot_bnd]>threshold]
            y = y_grid[field[plot_bnd]>threshold]
            z = z_grid[field[plot_bnd]>threshold]
            alpha=field[plot_bnd][field[plot_bnd]>threshold]**2 if len(field[plot_bnd][field[plot_bnd]>threshold]) else 1
            scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold],
                                s=density_scatter_size,alpha=transp_fld, cmap='viridis')
        if not remove_colorbar: plt.colorbar(scatter)
    elif plot_bnd != -1:
        if restrict_mol:
            plot_inds =  (x_grid > min_[0]) * (x_grid < max_[0]) * (y_grid > min_[1]) * (y_grid < max_[1]) * (z_grid > min_[2]) * (z_grid < max_[2])
            plot_inds = (x_grid < 1000)
            if threshold is None: 
                threshold = 0.1 * np.max(field[plot_bnd][plot_inds])
            plot_inds = plot_inds * (field[plot_bnd]>threshold)

            # (field[plot_bnd]>threshold) *
            x = x_grid[plot_inds]
            y = y_grid[plot_inds]
            z = z_grid[plot_inds]
            # alpha=field[plot_bnd][plot_inds]**2 if len(field[plot_bnd][plot_inds]) else 1
            scatter =ax.scatter(x,y,z,c=field[plot_bnd][plot_inds], s=density_scatter_size)
        else:
            x = x_grid[field[plot_bnd]>threshold]
            y=  y_grid[field[plot_bnd]>threshold]
            z = z_grid[field[plot_bnd]>threshold]
            alpha=field[plot_bnd][field[plot_bnd]>threshold]**2 if len(field[plot_bnd][field[plot_bnd]>threshold]) else 1
            size = alpha * 50
            scatter =ax.scatter(x,y,z,c=field[plot_bnd][field[plot_bnd]>threshold], s=density_scatter_size)
        if not remove_colorbar: 
            cbar = plt.colorbar(scatter)
            cbar.ax.tick_params(labelsize=16)
    ax.set_xlim3d(np.min(atm_pos,axis=0)[0], np.max(atm_pos,axis=0)[0])
    ax.set_ylim3d(np.min(atm_pos,axis=0)[1], np.max(atm_pos,axis=0)[1])
    ax.set_zlim3d(np.min(atm_pos,axis=0)[2]-0.5, np.max(atm_pos,axis=0)[2]+0.5)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # for bnd in candidaten_bnds:
    timestep_fn_save = "" if timestep is None else "x_{}".format(timestep)
    timestep = "" if timestep is None else ". $x_{{{}}}$".format(timestep)
    if title == -1: pass
    elif atms_required is not None: 
        bin_class = atms_required[batch_ind]
        bin = "{} < n <= {}".format(bins[bin_class-1], bins[bin_class])
        plt.title(plt.title("Moleule {} failed. Required atom number {}".format(batch_ind,bin)), fontsize=7)
    elif title is None: plt.title("Moleule {} failed".format(batch_ind) + timestep, fontsize=7)
    else: plt.title(title, fontsize=7)
    if save_fig_name:
        plt.legend(fontsize=16)
        plt.savefig(save_fig_name)
    elif show_plot: plt.show()
    return ax, scatter
    # plt.savefig("/home/alex/Desktop/molecule_{}".format(batch_ind)+ timestep_fn_save+ ".png")


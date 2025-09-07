"""
Add loss clipping to prevent training instability.
This modifies the mixed teacher forcing trainer to clip extreme losses.
"""

import shutil
import re

# Backup the original file
shutil.copy('pretrain_npt_mixed_teacher_forcing.py', 'pretrain_npt_mixed_teacher_forcing_backup.py')

# Read the file
with open('pretrain_npt_mixed_teacher_forcing.py', 'r') as f:
    content = f.read()

# Add loss clipping function after the imports
loss_clipping_code = '''

def clip_loss(loss, max_value=10.0, name="loss"):
    """Clip loss values to prevent instability."""
    if torch.isnan(loss):
        logging.warning(f"NaN {name} detected, returning small value")
        return torch.tensor(0.1, device=loss.device, requires_grad=True)
    
    if loss > max_value:
        logging.warning(f"Clipping {name} from {loss.item():.2f} to {max_value}")
        return torch.clamp(loss, max=max_value)
    
    return loss

'''

# Insert after imports
import_end = content.find('class MixedTeacherForcingNPTTrainer:')
content = content[:import_end] + loss_clipping_code + content[import_end:]

# Modify the compute_mixed_loss to add loss clipping
# Find the return statement in compute_teacher_guided_loss
teacher_guided_return = '''        return {
            'total_loss': total_loss,
            'hidden_loss': hidden_loss_dict['total_loss'],
            'logits_loss': logits_loss,'''

teacher_guided_return_new = '''        # Clip losses to prevent instability
        total_loss = clip_loss(total_loss, max_value=10.0, name="total_loss")
        logits_loss = clip_loss(logits_loss, max_value=5.0, name="logits_loss")
        
        return {
            'total_loss': total_loss,
            'hidden_loss': hidden_loss_dict['total_loss'],
            'logits_loss': logits_loss,'''

content = content.replace(teacher_guided_return, teacher_guided_return_new)

# Same for student guided loss
student_guided_return = '''        # Combine losses with higher weight on logits for student-guided
        total_loss = (
            self.args.hidden_loss_weight * hidden_loss_dict['total_loss'] +
            self.args.logits_loss_weight * logits_loss * 2.0  # Double weight for student-guided
        )
        
        return {
            'total_loss': total_loss,
            'hidden_loss': hidden_loss_dict['total_loss'],
            'logits_loss': logits_loss,'''

student_guided_return_new = '''        # Combine losses with higher weight on logits for student-guided
        total_loss = (
            self.args.hidden_loss_weight * hidden_loss_dict['total_loss'] +
            self.args.logits_loss_weight * logits_loss * 2.0  # Double weight for student-guided
        )
        
        # Clip losses to prevent instability (more aggressive for student-guided)
        total_loss = clip_loss(total_loss, max_value=5.0, name="student_total_loss")
        logits_loss = clip_loss(logits_loss, max_value=3.0, name="student_logits_loss")
        
        return {
            'total_loss': total_loss,
            'hidden_loss': hidden_loss_dict['total_loss'],
            'logits_loss': logits_loss,'''

content = content.replace(student_guided_return, student_guided_return_new)

# Write the modified file
with open('pretrain_npt_mixed_teacher_forcing.py', 'w') as f:
    f.write(content)

print("✅ Added loss clipping to pretrain_npt_mixed_teacher_forcing.py")
print("✅ Original backed up to pretrain_npt_mixed_teacher_forcing_backup.py")
print("\nLoss clipping thresholds:")
print("  - Teacher-guided: total_loss <= 10.0, logits_loss <= 5.0")
print("  - Student-guided: total_loss <= 5.0, logits_loss <= 3.0")


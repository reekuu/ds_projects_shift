import numpy as np


def ann_pay(i: float, n: int, S: float) -> float:
    '''
    Formula of the annuity payment.

    Parameters:
        :param i : Monthly interest rate on a loan.
        :param n : Number of loan repayment periods.
        :param S : Loan ammount.
    Returns:
        res : Returns the amount of the annuity payment.    
    '''
    annuity = S * i * (i + 1)**n / ((i + 1)**n - 1)

    return annuity


def df_loss(i: float, n: int, S: float, A: float) -> float:
    '''
    Formula of the loss function.

    We represent the function as the square of the difference between
    the payment formula and the actual size of the payment.

    Formula is generated using Sympy.

    f = (S * i * (i + 1)**n / ((i + 1)**n - 1) - A)**2
    loss = derivative_f = f.diff(i)

    Parameters:
        :param i : Monthly interest rate on a loan.
        :param n : Number of loan repayment periods.
        :param S : Loan ammount.
        :param A : Annuity payment.
    Returns:    
        res : Returns the derivative of a compound loss function.
    '''
    loss = (-A + S*i*(i + 1)**n/((i + 1)**n - 1))*(-2*S*i*n*(i + 1)**(2*n)/((i + 1)*((i + 1)**n - 1)**2)
        + 2*S*i*n*(i + 1)**n/((i + 1)*((i + 1)**n - 1)) + 2*S*(i + 1)**n/((i + 1)**n - 1))
    
    return loss


def gradient_descent(S: float, A: float, n: int, lr: float, iter_num: int) -> float:
    '''
    Formula of the Gradient Descent.
    
    Parameters:
        :param S : Loan ammount.
        :param A : Annuity payment.
        :param n : Number of loan repayment periods.
        :param lr : Learning rate, step size of Gradient Descent.
        :param iter_num : Number of Gradient Descent iterations.
    Returns:
        res : Returns the local minimum of the loss function.
    '''
    # Start point of Gradient Descent
    x = 1
    
    # Calculate Gradient Descent
    for i in range(iter_num):
        x = x - lr * df_loss(x, n, 1, A)

    return x


def interest_calculator(S: float, A: float, lr: float=0.5, iter_num: int=100) -> float:
    '''
    Calculates loan interest rate using Gradient Descent.
    
    Parameters:
        :param S : Loan ammount.
        :param A : Annuity payment.
        :param lr : Learning rate, step size of Gradient Descent.
        :param iter_num : Number of Gradient Descent iterations.
    Returns:
        res : Returns loan interest rate.
    '''
    result = np.nan

    try:
        # Normalized annuity size
        ANorm = A / S

        # Lower and upper edges of iteration
        five_years = 60
        lower_edge = int(np.floor(S / A))
        upper_edge = lower_edge + five_years

        # Set interest rate bounds
        min_interest = 0.05
        max_interest = 0.8

        # Iterating over the number of payment months and selecting the interest rate
        for n in range(lower_edge, upper_edge):
            interest = gradient_descent(1, ANorm, n, lr, iter_num)
            if abs(ann_pay(interest, n, S)) - A == 0:
                result = round(interest * 12, 5)
                result = np.clip(result, min_interest, max_interest)
                break
    except:
        pass

    return result
    
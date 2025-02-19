from backtesting import Strategy

class SimpleFollowSignalsStrategy(Strategy):
    def init(self):
        self.signal = self.I(lambda: self.data.Signal)
        self.previous_signal = 0
        # self.size = 0.1

    def next(self):
        current_signal = self.signal[-1]

        if current_signal != self.previous_signal:
            if current_signal == 1:
                
                if self.position.is_short:
                    self.position.close()
                    
                if not self.position.is_long:
                    self.buy(size=1)
                    
            elif current_signal == -1:
                if self.position.is_long:
                    self.position.close()
                   
                if not self.position.is_short:
                    self.sell(size=1)
            
                    
            # elif current_signal == 0:
            #     if self.position:
            #         self.position.close()

        self.previous_signal = current_signal
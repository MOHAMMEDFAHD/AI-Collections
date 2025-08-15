<table>
  <thead>
    <tr>
      <th>Aspect</th>
      <th>FNN</th>
      <th>CNN</th>
      <th>RNN</th>
      <th>LLM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Primary Use</td>
      <td>Basic pattern recognition</td>
      <td>Image and video processing</td>
      <td>Sequential data (e.g., time series, text)</td>
      <td>Natural language understanding & generation</td>
    </tr>
    <tr>
      <td>Data Handling</td>
      <td>Fixed-size inputs</td>
      <td>Grid-like data (e.g., 2D images)</td>
      <td>Time-dependent sequences</td>
      <td>Textual data with context</td>
    </tr>
    <tr>
      <td>Key Feature</td>
      <td>Fully connected layers</td>
      <td>Convolutions for feature extraction</td>
      <td>Memory of previous inputs</td>
      <td>Transformer architecture</td>
    </tr>
    <tr>
      <td>Strength</td>
      <td>Simple structure, easy to implement</td>
      <td>High accuracy for visual tasks</td>
      <td>Captures sequential relationships</td>
      <td>Understanding complex language tasks</td>
    </tr>
    <tr>
      <td>Weakness</td>
      <td>Not ideal for complex patterns</td>
      <td>Struggles with sequential data</td>
      <td>Vanishing gradient problem</td>
      <td>High computational cost</td>
    </tr>
    <tr>
      <td>Common Applications</td>
      <td>Regression, classification</td>
      <td>Object detection, image recognition</td>
      <td>Language modeling, stock prediction</td>
      <td>Chatbots, summarization, translation</td>
    </tr>
  </tbody>



  <h3>📊 Comparison of Different Types of Fields with Data</h3>

<table>
  <thead>
    <tr>
      <th>Aspect</th>
      <th>Data Science</th>
      <th>Data Engineering</th>
      <th>Data Analysis</th>
      <th>Data Modeling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Primary Role</td>
      <td>Extract insights and build predictive models</td>
      <td>Design and maintain data pipelines</td>
      <td>Analyze data to inform decisions</td>
      <td>Define data structures and relationships</td>
    </tr>
    <tr>
      <td>Focus Area</td>
      <td>Machine learning, AI, statistics</td>
      <td>ETL, data warehouses, big data</td>
      <td>Visualizations, reporting, trends</td>
      <td>Schemas, normalization, database design</td>
    </tr>
    <tr>
      <td>Key Tools</td>
      <td>Python, R, TensorFlow, scikit-learn</td>
      <td>Spark, Hadoop, Apache Kafka</td>
      <td>Excel, Tableau, Power BI</td>
      <td>ERD tools, SQL, NoSQL design tools</td>
    </tr>
    <tr>
      <td>Output</td>
      <td>Models, insights, forecasts</td>
      <td>Clean, structured data</td>
      <td>Actionable insights, dashboards</td>
      <td>Efficient, scalable databases</td>
    </tr>
    <tr>
      <td>Challenges</td>
      <td>Complexity of models, interpretability</td>
      <td>Handling large data at scale</td>
      <td>Misinterpretation of data</td>
      <td>Designing for flexibility and efficiency</td>
    </tr>
    <tr>
      <td>Common Applications</td>
      <td>Recommendation systems, fraud detection</td>
      <td>Building data pipelines for ML models</td>
      <td>Market trends, customer segmentation</td>
      <td>Database design for e-commerce, finance</td>
    </tr>
  </tbody>
</table>

</table>

<h3>📊 Comparison of Different Types of Loss Functions of Regression Models</h3>

<table>
  <thead>
    <tr>
      <th>Aspect</th>
      <th>Mean Squared Error (MSE)</th>
      <th>Mean Absolute Error (MAE)</th>
      <th>Root Mean Squared Error (RMSE)</th>
      <th>R² (Coefficient of Determination)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Definition</td>
      <td>Average of squared differences between predicted and actual values</td>
      <td>Average of absolute differences between predicted and actual values</td>
      <td>Square root of the mean squared error</td>
      <td>Proportion of variance in the dependent variable explained by the model</td>
    </tr>
    <tr>
      <td>Formula</td>
      <td>$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2 $$</td>
      <td>$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_{\text{true}, i} - y_{\text{pred}, i}| $$</td>
      <td>$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2} $$</td>
      <td>$$ R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} $$</td>
    </tr>
    <tr>
      <td>Output Range</td>
      <td>0 to infinity</td>
      <td>0 to infinity</td>
      <td>0 to infinity</td>
      <td>-∞ to 1</td>
    </tr>
    <tr>
      <td>Sensitivity</td>
      <td>Penalizes larger errors more due to squaring</td>
      <td>Treats all errors equally</td>
      <td>Similar to MSE but in the same units as the data</td>
      <td>Sensitive to overfitting and underfitting</td>
    </tr>
    <tr>
      <td>Use Case</td>
      <td>Regression tasks where large errors are critical</td>
      <td>Robust regression tasks with outliers</td>
      <td>When interpretability in original units is needed</td>
      <td>Model evaluation and variance explanation</td>
    </tr>
    <tr>
      <td>Interpretation</td>
      <td>Lower is better; higher indicates poor fit</td>
      <td>Lower is better; higher indicates poor fit</td>
      <td>Lower is better; higher indicates poor fit</td>
      <td>Closer to 1 is better; negative values indicate poor fit</td>
    </tr>
  </tbody>
</table>



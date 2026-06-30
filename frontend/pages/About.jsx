const About = () => {
  return (
    <div className="about">
      <h1>About Kuzushiji Reader</h1>
      <p>
        Kuzushiji Reader is a web application designed to transcribe and translate kuzushiji manuscripts. It allows users to upload images of kuzushiji texts, which are then processed to extract the characters and provide translations.
      </p>
      <h2>Features</h2>
      <ul>
        <li>Image upload for kuzushiji manuscripts</li>
        <li>Character recognition and transcription</li>
        <li>Translation of transcribed text</li>
        <li>User-friendly interface for easy navigation</li>
      </ul>
      <h2>Technology Stack</h2>
      <p>
        The application is built using React for the frontend, with a backend powered by Python and machine learning models for character recognition. The system leverages convolutional neural networks (CNNs) to accurately identify kuzushiji characters from images.
      </p>
      <h2>Contributing</h2>
      <p>
        Contributions to the project are welcome! If you have ideas for new features, improvements, or bug fixes, please feel free to submit a pull request on our GitHub repository.
      </p>
    </div>
  )
}

export default About